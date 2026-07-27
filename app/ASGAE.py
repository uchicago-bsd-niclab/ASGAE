import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch_geometric.nn import ChebConv, GraphNorm, Linear, BatchNorm
import torch_geometric as tog
from torch_geometric.utils import to_dense_batch
import app.util as utl
import numpy as np
import copy
from app.custom_chamfer import chamfer_distance
from app.DataLoaders import MyData, LoadedDataset, GetDataLoaders
from tqdm import tqdm
        

class RegistMesh(nn.Module):
    def __init__(self, in_channel, bias=True):
        super(RegistMesh, self).__init__()
        self.in_channel = in_channel
        self.updateTemp = False
        
    def forward(self, x, scaling=False):
        x=x.detach()
        B, C, N = x.shape
        x = x.permute(0, 2, 1)  # [B, N, 3]
        if not hasattr(self, 'mean'):
            self.template = x.mean(dim=0).detach()  # [1, 512, 3]
            self.template -= self.template.mean(dim=0).unsqueeze(0)  # [1,1,3] Center the template
            # self.template += self.center.to(self.template).unsqueeze(0)  # Center the template to a specific point
        else:
            self.template = self.mean.to(x)#.unsqueeze(0).expand(B, -1, -1)
        template = self.template.unsqueeze(0).expand(B, -1, -1)  # [B, N, 3]

        x_mean = x.mean(dim=1, keepdim=True)       # [B, 1, 3]
        t_mean = template.mean(dim=1, keepdim=True)

        x_centered = x - x_mean
        t_centered = template - t_mean

        H = x_centered.transpose(1, 2) @ t_centered
        U, _, Vt = torch.linalg.svd(H)
        R = Vt.transpose(2, 1) @ U.transpose(2, 1)  # [B, 3, 3]

        # Fix reflections (NO inplace)
        detR = torch.det(R)
        if (detR < 0).any():
            Vt[detR < 0, -1, :] *= -1
            R = Vt.transpose(2, 1) @ U.transpose(2, 1)  # [B, 3, 3]

        # Scaling estimation
        if scaling:
            # Convert tensors to numpy for scaling calculation
            AA = (R@x_centered.transpose(2, 1)).transpose(2,1)
            BB = t_centered
            s = (torch.mean(torch.linalg.norm(BB, axis=2),dim=1) / (torch.mean(torch.linalg.norm(AA, axis=2),dim=1)+1e-8))
            R = s.unsqueeze(1).unsqueeze(2)*R
        
        # Translation
        t = t_mean.transpose(2, 1) - R @ x_mean.transpose(2, 1) # [B, 3]

        # Build transformation matrix
        tmat = torch.eye(4, device=x.device).unsqueeze(0).repeat(B, 1, 1)
        tmat[:, :3, :3] = R  # [B, 3, 3]
        tmat[:, :3, 3] = t.squeeze(2)

        # Apply transformation
        x_aligned = utl.apply_transform(x, tmat)

        self.transformed = x_aligned.detach()
        self.tmatrix = tmat.detach()
        self.R = R.detach()
        self.t = t.detach()
        self.template_centered = t_centered.detach()
        if scaling:
            self.update_template(self.transformed.detach()/s.unsqueeze(1).unsqueeze(2))
        else:
            self.update_template(self.transformed.detach())
            
        return self.tmatrix.to(torch.float32).detach()
    
    def update_template(self, new_sample, alpha=0.1):
        if hasattr(self, 'mean') and self.updateTemp:
            vaux = self.mean.clone().detach()
            self.mean = alpha * new_sample.mean(0).detach() + (1 - alpha) * vaux.to(new_sample)
        else:
            self.mean = new_sample.mean(0).detach()
            # self.updateTemp = True


class ChebLayer(nn.Module):
    def __init__(self, in_channels, out_channels, K = 4, bias=True):
        super(ChebLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.chebconv = ChebConv(in_channels, out_channels, K=K, bias=bias)
        self.Gbn = GraphNorm(out_channels)
        self.bn = BatchNorm(out_channels)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
    def forward(self, x, edge_index, edge_weight=None, batch=None, normMode='Graph'):
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=x.device, dtype=x.dtype)
        x = self.chebconv(x, edge_index, edge_weight=edge_weight.to(torch.float32), batch=batch)
        x = self.relu(x)
        if normMode == 'Graph':
            x = self.Gbn(x, batch=batch)
        elif normMode == 'Batch':
            x = self.bn(x)
        return x  # Return the output of the Chebyshev layer after activation and normalization

class GraphEncoder(nn.Module):
    def __init__(self, in_channels, latent_size, drop_prob=0.1, bias=True):
        super(GraphEncoder, self).__init__()
        self.in_channel = in_channels
        self.latent_size = latent_size
        hidden_features = self.latent_size
        Ks = 4
        self.R = tog.nn.Sequential('x, edge_index, edge_weight, batch', [
            (BatchNorm(in_channels), 'x -> x'),
            (ChebLayer(in_channels, hidden_features // 16, K=Ks, bias=bias), 'x, edge_index, edge_weight, batch -> x'),
            (ChebLayer(hidden_features // 16, hidden_features // 8, K=Ks, bias=bias), 'x, edge_index, edge_weight, batch -> x'),
            (ChebLayer(hidden_features // 8, hidden_features // 4, K=Ks, bias=bias), 'x, edge_index, edge_weight, batch -> x'),
            (ChebLayer(hidden_features // 4, hidden_features // 2, K=Ks, bias=bias), 'x, edge_index, edge_weight, batch -> x'),
        ])
        self.Rlin = Linear(hidden_features // 2, hidden_features, bias=bias)
    def forward(self, src, eval=False):         
        position, features, batch, edge_indices, edge_weights = src.pos, src.x, src.batch, src.edge_index, src.edge_weight
        # edge_indices, edge_weights = remove_self_loops(edge_indices, edge_weights)
        if not eval:
            pos1,_ = utl.applyTransformGraphPos(position, src.batch)#, scale=False)
            pos2,_ = utl.applyTransformGraphPos(position, src.batch)#, scale=False)
            x_input1 = torch.concat([pos1.to(torch.float32), features.to(torch.float32)[:,3:]], dim=1)
            m_input1 = self.R(x_input1, edge_indices, edge_weights, batch)
            R_out1 = self.Rlin(m_input1)
            R1 = tog.utils.softmax(R_out1, batch, dim=0)
            
            x_input2 = torch.concat([pos2.to(torch.float32), features.to(torch.float32)[:,3:]], dim=1)
            m_input2 = self.R(x_input2, edge_indices, edge_weights, batch)
            R_out2 = self.Rlin(m_input2)
            R = tog.utils.softmax((R_out1+R_out2)/2, batch, dim=0)
            R2 = tog.utils.softmax(R_out2, batch, dim=0)
            R1a, mask = to_dense_batch(R, batch)

            U =to_dense_batch(x_input1, batch)[0]
            L = torch.permute(R1a, (0, 2, 1))@U
            del pos1, pos2, m_input1, R_out2, R2,x_input2, R1a, R1, R_out1, features, position, batch, edge_indices, edge_weights, R, U
        else:
            pos1 = position
            pos2 = position
            Tmat = torch.eye(4, device=position.device).unsqueeze(0).repeat(len(src), 1, 1)
            x_input1 = torch.concat([pos1.to(torch.float32), features.to(torch.float32)[:,3:]], dim=1)
            m_input2 = self.R(x_input1, edge_indices, edge_weights, batch)
            R_out1 = self.Rlin(m_input2)
            R1 = tog.utils.softmax(R_out1, batch, dim=0)
            R1a, mask = to_dense_batch(R1, batch)
            U =to_dense_batch(x_input1, batch)[0]
            
            if len(U.shape)==2:
                R1a = R1a.unsqueeze(0)
                U = U.unsqueeze(0)
            L = torch.permute(R1a, (0, 2, 1))@U
            # kl_mean = 0
            del pos1, pos2, edge_indices, edge_weights, R_out1, R1a, R1, features, position, batch, U

        return L, mask, x_input1, m_input2#, kl_mean
    
class GraphDecoder(nn.Module):
    def __init__(self, in_channels, latent_size, drop_prob=0.1, bias=True):
        super(GraphDecoder, self).__init__()
        self.in_channel = in_channels
        self.latent_size = latent_size
        hidden_features = self.latent_size
        Ks = 4
        self.M = tog.nn.Sequential('x, edge_index, edge_weight, batch', [
            (ChebLayer(hidden_features // 2, hidden_features // 2, K=Ks, bias=bias), 'x, edge_index, edge_weight, batch -> x'),
            (Linear(hidden_features // 2, hidden_features, bias=bias), 'x -> x'),
            (nn.Softmax(dim=1), 'x -> x')
        ])
    def forward(self, src, x_input):         
        MInv= self.M(x_input, src.edge_index, src.edge_weight, src.batch)
        M = to_dense_batch(MInv, src.batch)[0]
        return M


class AE(nn.Module):
    def __init__(self, in_channels, latent_size, bias=True, drop_prob= 0.7, Dataset=None, trainDic={}):
        super(AE, self).__init__()
        self.in_channels=in_channels
        self.latent_size = latent_size
        self.Encoder = GraphEncoder(self.in_channels, self.latent_size, drop_prob=drop_prob,bias=bias)
        self.Decoder = GraphDecoder(self.in_channels, self.latent_size, drop_prob=drop_prob,bias=bias)
        
        self.Outcoord = tog.nn.Sequential('x, edge_index, edge_weight, batch', [
            (ChebLayer(latent_size+in_channels, int(latent_size), K=4, bias=bias), 'x, edge_index, edge_weight, batch -> x'),
            (Linear(int(latent_size), int(latent_size),bias=bias), 'x -> x'),
            (nn.Tanh(), 'x -> x')
        ])
        self.criterion = nn.MSELoss()
        self.reg = RegistMesh(3)
        self.scalCoord = 0.0001
        
        dataset = LoadedDataset(Dataset,[])
        self.train_loader, self.val_loader, self.test_loader = GetDataLoaders(dataset, shuffle=trainDic['shuffle'], batch_size=trainDic['batch_size'], batchtest=trainDic['test_batch_size'],train_set_percentage=trainDic['setSplit'][0], val_set_percentage=trainDic['setSplit'][1])

    def forward(self, target):
        # target = MyData(x=x, pos=pos, edge_index=edge_index, edge_weight=edge_weight, batch=batch)
        self.trg=target.clone()
        del target
        self.L0, self.mask2, self.U, inputT = self.Encoder(self.trg)
        self.M= self.Decoder(self.trg, inputT)
        torch.cuda.empty_cache()
        
        coord = self.L0[:,:,:3].permute(0,2,1) # [8, 3, 512]
        color = self.L0[:,:,3:].permute(0,2,1) # [8, 3, 512]
        
        T_est = self.reg(coord)

        coordR1 = utl.apply_transform(coord.permute(0,2,1), T_est).permute(0,2,1)
        self.L1 = torch.cat((coordR1, color), dim=1).permute(0,2,1)
        
        self.Out =  self.ObtainOutput(self.L0, self.M, self.mask2, self.trg) 
        return self.Out, self.U
    
    def evaluate(self, target):
        # target = MyData(x=x, pos=pos, edge_index=edge_index, edge_weight=edge_weight, batch=batch)
        self.trg=target.clone()
        del target
        self.L0, self.mask2, self.U, inputT = self.Encoder(self.trg, eval=True)
        self.M = self.Decoder(self.trg, inputT)
        
        coord = self.L0[:,:,:3].permute(0,2,1) # [8, 3, 512]
        color = self.L0[:,:,3:].permute(0,2,1) # [8, 3, 512]
        
        T_est = self.reg(coord)
        
        coordR1 = utl.apply_transform(coord.permute(0,2,1), T_est).permute(0,2,1)
        self.L1 = torch.cat((coordR1, color), dim=1).permute(0,2,1)
        
        self.Out = self.ObtainOutput(self.L0, self.M, self.mask2, self.trg)
        return self.Out, self.U
    
    def ObtainOutput(self, L0, M, mask, target):
        xLOini = M@L0
        if not M.size(0) == 1:
            xLOini[~mask] = 0
        xLO = utl.from_dense_batch(xLOini, mask)[0]
        MDense = utl.from_dense_batch(M, mask)[0]
        
        postM = self.scalCoord*(self.Outcoord(torch.cat([MDense, xLO], dim=1), target.edge_index,target.edge_weight.to(torch.float32), batch=target.batch))+MDense
        Out = utl.from_dense_batch(to_dense_batch(postM, target.batch)[0]@L0, mask)[0]
        del postM, xLO, MDense, xLOini
        torch.cuda.empty_cache()
        return Out
      
    def save_mesh(self, folderName, epoch, save_src=False):
        if not os.path.exists(folderName):
            os.makedirs(folderName)
        target = copy.deepcopy(self.trg)
        target.pos = self.U[:,:3]
        target.x[:,3:] = self.U[:,3:]
        target_mesh = utl.GenerateMesh(self.trg[0])
        utl.WritePolyData(target_mesh, os.path.join(folderName, 'Original_Mesh_Trg.vtp'))
        
        xLT1 = utl.from_dense_batch(self.M@self.L0, self.mask2)[0]
        
        target = copy.deepcopy(self.trg)
        target.pos=self.Out[:,:3]
        target.x[:,3:]=self.Out[:,3:]
        trg_mesh = utl.GenerateMesh(target[0])
        disttrg = utl.ComputePointToSurfaceError(target_mesh, target[0].pos.detach().cpu().numpy())
        res_trg = utl.ComputePointToSurfaceTextureError(target_mesh,target[0].pos.detach().cpu().numpy(),target[0].x[:,3:].detach().cpu().numpy(), metric='L2')
        # d1,d2,dt1,dt2= utl.compute_chamfer_distances_mesh(target_mesh,trg_mesh, textureDist=True)
        mesh_with_DisTex1 = utl.add_distance_scalar_to_mesh(target_mesh, disttrg, res_trg['per_point_error'], name='TextureError')
        mesh_with_DisTex2 = utl.add_distance_scalar_to_mesh(trg_mesh, disttrg, res_trg['per_point_error'], name='TextureError')
        utl.WritePolyData(mesh_with_DisTex2, os.path.join(folderName,'test_trg_'+str(epoch)+'.vtp'))
        utl.WritePolyData(mesh_with_DisTex1, os.path.join(folderName, 'Original_Mesh_Trg.vtp'))
        target.pos=xLT1[:,:3]
        target.x[:,3:]=xLT1[:,3:]
        torch.save(target[0], os.path.join(folderName,'test_trg_BFL'+str(epoch)+'.pt'))
        
        utl.SaveLandmarks(self.L0[0].cpu().numpy(), os.path.join(folderName,'LatentPoints'+str(epoch)+'.vtp'))  
        # save latent template
        utl.SaveLandmarks(self.reg.mean.cpu().numpy(), os.path.join(folderName,'Template.vtp'))
        del target, trg_mesh
        
        
    
    def transfer(self, src, trg, epoch, fileName):
        source = copy.deepcopy(src)
        target = copy.deepcopy(trg)
        del src, trg
        L0O, mask1, UO, x_norm1 = self.Encoder(source, eval=True)
        MO= self.Decoder(source, x_norm1)
        L0T, mask2, UT, x_norm2 = self.Encoder(target, eval=True)
        MT= self.Decoder(target, x_norm2)
        
        
        coordTO = L0O[:,:,:3].permute(0,2,1) # [8, 3, 512]
        colorTO = L0O[:,:,3:].permute(0,2,1) # [8, 3, 512]
        coordTT = L0T[:,:,:3].permute(0,2,1) # [8, 3, 512]
        colorTT = L0T[:,:,3:].permute(0,2,1) # [8, 3, 512]

        TO_est = self.reg(coordTO)
        TT_est = self.reg(coordTT)
        
        coordR1TO = utl.apply_transform(coordTO.permute(0,2,1), TO_est).permute(0,2,1)
        coordR1TT = utl.apply_transform(coordTT.permute(0,2,1), TT_est).permute(0,2,1)
        
        L1_TO = torch.cat(((coordR1TO), (colorTO)), dim=1).permute(0,2,1)
        L1_TT = torch.cat(((coordR1TT), (colorTT)), dim=1).permute(0,2,1)
        
        xRO = self.ObtainOutput(L0T, MO, mask1, source)
        xLT = self.ObtainOutput(L0O, MT, mask2, target)
        
        if not os.path.exists(fileName):
            os.makedirs(fileName)
        if not os.path.exists(os.path.join(fileName,'Original_Transfer_Trg.pt')):
            torch.save(target, os.path.join(fileName,'Original_Transfer_Trg.pt'))
        if not os.path.exists(os.path.join(fileName,'Original_Transfer_Src.pt')):
            torch.save(source, os.path.join(fileName,'Original_Transfer_Src.pt'))
        
        if not MO.size(0) == 1:
            UO[~mask1] = 0
            UT[~mask2] = 0
            UO= utl.from_dense_batch(UO, mask1)[0]
            UT= utl.from_dense_batch(UT, mask2)[0]
        
        source.pos=UO
        target.pos=UT
        torch.save(source, os.path.join(fileName,'Original_source_src_'+str(epoch)+'.pt'))
        torch.save(target, os.path.join(fileName,'Original_source_trg_'+str(epoch)+'.pt'))
        
        source.pos=xRO[:,:3]
        target.pos=xLT[:,:3]
        source.x[:,3:]=xRO[:,3:]
        target.x[:,3:]=xLT[:,3:]
        torch.save(source, os.path.join(fileName,'test_transfer_src_'+str(epoch)+'.pt'))
        torch.save(target, os.path.join(fileName,'test_transfer_trg_'+str(epoch)+'.pt'))
        
        utl.SaveLandmarks(L0O[0].cpu().numpy(), os.path.join(fileName,'LatentPointsSource'+str(epoch)+'.vtp'))
        utl.SaveLandmarks(L0T[0].cpu().numpy(), os.path.join(fileName,'LatentPointsTarget'+str(epoch)+'.vtp'))
        del source, target, xRO, xLT
        
    def lossChamfer(self):
        batchSize = self.trg.num_graphs
        shiftedIndex = (torch.arange(batchSize) + 1)%batchSize
        shiftedBatch = tog.data.Batch.from_data_list(self.trg[shiftedIndex])

        Y, mask = to_dense_batch(self.U, self.trg.batch)

        Y_trans = self.ObtainOutput(L0=self.L0,
                                    M=self.M[shiftedIndex],
                                    mask=mask[shiftedIndex],
                                    target=shiftedBatch)
        Y_trans = to_dense_batch(Y_trans, shiftedBatch.batch)[0]

        chamfer_transfer_position, chamfer_transfer_texture = chamfer_distance(
            x=Y_trans[..., :3],
            y=Y[..., :3],
            x_lengths=mask[shiftedIndex].sum(1),
            y_lengths=mask.sum(1),
            x_texture=Y_trans[..., 3:],
            y_texture=Y[..., 3:]
        )

        return chamfer_transfer_position, chamfer_transfer_texture
            

    def loss(self, type_loss='Full', verbose=False):
        U =to_dense_batch(self.U, self.trg.batch)[0]
        if type_loss=='Full':
            loss_coord = self.criterion(self.U[:,:3], self.Out[:,:3])
            loss_color = self.criterion(self.U[:,3:], self.Out[:,3:])
            
            loss_latent_position, loss_latent_texture = chamfer_distance(
                x=U[..., :3],
                y=self.L0[:, :, :3],
                x_lengths=self.mask2.sum(1),
                #y_lengths=self.latent_size,
                x_texture=U[:,:, 3:],
                y_texture=self.L0[:,:, 3:]
            )

            transfer_loss=self.lossChamfer()
            
            loss_crd = loss_coord + 0.1*loss_latent_position +transfer_loss[0]
            loss_clr = loss_color + 0.1*loss_latent_texture +transfer_loss[1]
            
            loss = loss_crd + 1000*loss_clr
            if verbose:
                print(f"\r\nLoss Coord: \t\t\tMSE: {loss_coord.item():.4f}, \tChamfer-GTLat: {loss_latent_position.item():.4f}, \tChamfer-Transfer: {transfer_loss[0].item():.4f}. \
                      \nLoss Texture (factor=1000): \tMSE: {loss_color.item():.4f}, \t\tChamfer-GTLat: {loss_latent_texture.item():.4f}, \t\tChamfer-Transfer: {transfer_loss[1].item():.4f}.", end='', flush=True)
        elif type_loss=='baseLat':
            loss_coord = self.criterion(self.U[:,:3], self.Out[:,:3])
            loss_color = self.criterion(self.U[:,3:], self.Out[:,3:])
            
            distLYcoord = chamfer_distance(U[:,:,:3], self.L0[:,:,:3], x_lengths=self.mask2.sum(1))[0]
            distLYcolor = chamfer_distance(U[:,:,3:], self.L0[:,:,3:], x_lengths=self.mask2.sum(1))[0]
            
            loss_crd = loss_coord + 0.1*distLYcoord 
            loss_clr = loss_color + 0.1*distLYcolor 
            
            loss = loss_crd + 1000*loss_clr 
        elif type_loss=='baseTrans':
            loss_coord = self.criterion(self.U[:,:3], self.Out[:,:3])
            loss_color = self.criterion(self.U[:,3:], self.Out[:,3:])
            
            loss_crd = loss_coord #+ 0.1*distLYcoord# - 0.1*disLossLLcoord
            loss_clr = loss_color #+ 0.1*distLYcolor# - 0.1*disLossLLcolor
            
            loss = loss_crd + 1000*loss_clr + self.lossChamfer()
        elif type_loss=='Baseline':
            loss_coord = self.criterion(self.U[:,:3], self.Out[:,:3])
            loss_color = self.criterion(self.U[:,3:], self.Out[:,3:])
            
            loss_crd = loss_coord
            loss_clr = loss_color
            
            loss = loss_crd + 1000*loss_clr
        return loss
    
    def point_to_surface_metric(self, fileName, source, target, saveData=False):
        """
        Estimates the point-to-surface metric between transferred source and target meshes.
        Uses the transferred graphs from the transfer function.
        Returns the average minimum distance from each point in the transferred source to the surface of the transferred target.
        """
        source_copy = copy.deepcopy(source)
        target_copy = copy.deepcopy(target)
        src = copy.deepcopy(source)
        trg = copy.deepcopy(target)
        # Run transfer to get transferred source and target
        L0O, mask1, UO, x_norm1 = self.Encoder(source, eval=True)
        MO = self.Decoder(source, x_norm1)
        L0T, mask2, UT, x_norm2 = self.Encoder(target, eval=True)
        MT = self.Decoder(target, x_norm2)
        
        MatrixRprimeO = MO @ L0T
        MatrixLprimeO = MT @ L0O
        if not MO.size(0) == 1:
            MatrixRprimeO[~mask1] = 0
            MatrixLprimeO[~mask2] = 0
        
        xRO = self.ObtainOutput(L0T,MO, mask1,source)
        
        xLT = self.ObtainOutput(L0O, MT, mask2, target)
        
        source_mesh = utl.GenerateMesh(source_copy)
        target_mesh = utl.GenerateMesh(target_copy)

        # Calculate the point-to-surface distance
        source_points = xLT[:,:3].cpu().numpy()
        target_points = xRO[:,:3].cpu().numpy()
        source_texture = xLT[:,3:].cpu().numpy()
        target_texture = xRO[:,3:].cpu().numpy()

        distSrc = utl.ComputePointToSurfaceError(source_mesh, source_points)
        distTrg = utl.ComputePointToSurfaceError(target_mesh, target_points)
        src.pos=xRO[:,:3]
        trg.pos=xLT[:,:3]
        src.x[:,3:]=xRO[:,3:]
        trg.x[:,3:]=xLT[:,3:]
        src_mesh = utl.GenerateMesh(src)
        trg_mesh = utl.GenerateMesh(trg)
        res_src = utl.ComputePointToSurfaceTextureError(source_mesh,source_points,source_texture, metric='L2')
        res_trg = utl.ComputePointToSurfaceTextureError(target_mesh,target_points,target_texture, metric='L2')
        # d1,d2,dt1,dt2= utl.compute_chamfer_distances_mesh(target_mesh,src_mesh, textureDist=True)
        mesh_with_DisTex1 = utl.add_distance_scalar_to_mesh(target_mesh, distSrc, res_src['per_point_error'], name='TextureError')
        trg_mesh = utl.add_distance_scalar_to_mesh(trg_mesh, distSrc, res_src['per_point_error'], name='TextureError')
        mesh_with_DisTex2 = utl.add_distance_scalar_to_mesh(source_mesh, distTrg, res_trg['per_point_error'], name='TextureError')
        src_mesh = utl.add_distance_scalar_to_mesh(src_mesh, distTrg, res_trg['per_point_error'], name='TextureError')
        if saveData:
            utl.WritePolyData(src_mesh, os.path.join(fileName, 'src_mesh.vtp'))
            utl.WritePolyData(trg_mesh, os.path.join(fileName, 'trg_mesh.vtp'))
            utl.WritePolyData(mesh_with_DisTex1, os.path.join(fileName, 'distances_target_mesh.vtp'))
            utl.WritePolyData(mesh_with_DisTex2, os.path.join(fileName, 'distances_source_mesh.vtp'))
        
        return np.mean(distSrc), np.mean(distTrg), res_src['summary']['MSE'], res_trg['summary']['MSE']


    def Train_one_epoch(self, optimizer, device, type_loss='Full', verbose=False):
        self.train()
        total_loss = 0
        mse_coord = 0
        mae_coord = 0
        mse_color = 0
        mae_color = 0
        num_examples=0
        for data in tqdm(self.train_loader):
            batch_size = len(data)
            num_examples += batch_size
            data = data.to(device)
            optimizer.zero_grad()
            output, U = self.forward(data)
            loss = self.loss(type_loss=type_loss, verbose=verbose)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()* batch_size
            try:
                mse_coord_value = torch.mean((output[:,:3] - U[:,:3].to(torch.float32)) ** 2).item()
                mae_coord_value = torch.mean(torch.abs(output[:,:3] - U[:,:3].to(torch.float32))).item()
                mse_color_value = torch.mean((U[:,3:].to(torch.float32) - output[:,3:].to(torch.float32)) ** 2).item()
                mae_color_value = torch.mean(torch.abs(U[:,3:].to(torch.float32) - output[:,3:].to(torch.float32))).item()

                mse_coord += mse_coord_value * batch_size
                mae_coord += mae_coord_value * batch_size
                mse_color += mse_color_value * batch_size
                mae_color += mae_color_value * batch_size
            except: # in partial point cloud case
                mse_coord += 0
                mae_coord += 0
                mse_color += 0
                mae_color += 0
            del data, output, U
            torch.cuda.empty_cache()
        
        print('\n==TRAIN==')
        print('Loss: %f || MSE: %f, MAE: %f  || Texture --> MSE: %f, MAE: %f'
                        % (total_loss * 1.0 / num_examples, mse_coord * 1.0 / num_examples, mae_coord * 1.0 / num_examples, \
                    mse_color * 1.0 / num_examples, mae_color * 1.0 / num_examples))
        return total_loss * 1.0 / num_examples

    def Test_one_epoch(self, device, type_loss='Full', verbose=False):
        self.eval()
        total_loss = 0
        mse_coord = 0
        mae_coord = 0
        mse_color = 0
        mae_color = 0
        num_examples = 0
        with torch.no_grad():
            for data in tqdm(self.val_loader):
                batch_size = len(data)
                num_examples += batch_size
                data = data.to(device)
                output, U = self.evaluate(data)
                loss = self.loss(type_loss=type_loss, verbose=verbose)
                total_loss += loss.item() * batch_size
                try:
                    mse_coord_value = torch.mean((output[:,:3] - U[:,:3].to(torch.float32)) ** 2).item()
                    mae_coord_value = torch.mean(torch.abs(output[:,:3] - U[:,:3].to(torch.float32))).item()
                    mse_color_value = torch.mean((U[:,3:].to(torch.float32) - output[:,3:].to(torch.float32)) ** 2).item()
                    mae_color_value = torch.mean(torch.abs(U[:,3:].to(torch.float32) - output[:,3:].to(torch.float32))).item()

                    mse_coord += mse_coord_value * batch_size
                    mae_coord += mae_coord_value * batch_size
                    mse_color += mse_color_value * batch_size
                    mae_color += mae_color_value * batch_size
                except: # in partial point cloud case
                    mse_coord += 0
                    mae_coord += 0
                    mse_color += 0
                    mae_color += 0
                del data, output, U
                torch.cuda.empty_cache()
        
        print('==VAL==')
        print('Loss: %f || MSE: %f, MAE: %f  || Texture --> MSE: %f, MAE: %f'
                        % (total_loss * 1.0 / num_examples, mse_coord * 1.0 / num_examples, mae_coord * 1.0 / num_examples, \
                    mse_color * 1.0 / num_examples, mae_color * 1.0 / num_examples))
        return total_loss * 1.0 / num_examples