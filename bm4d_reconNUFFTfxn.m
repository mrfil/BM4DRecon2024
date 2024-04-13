% --------------------------------------------------------------------------------------------
%
%     Demo software for BM4D volumetric data reconstruction
%               Release ver. 3.2  (30 March 2015) - Adapted by Charles
%               Marchini 2024 to reconstruct data in a format that comes in
%               from an MRI scanner
%
% --------------------------------------------------------------------------------------------
%
% The software implements the iterative reconstruction algorithm described in the paper:
%
% M. Maggioni, V. Katkovnik, K. Egiazarian, A. Foi, "A Nonlocal Transform-Domain Filter 
% for Volumetric Data Denoising and Reconstruction", IEEE Trans. Image Process.,
% vol. 22, no. 1, pp. 119-133, January 2013. doi:10.1109/TIP.2012.2210725
%
% --------------------------------------------------------------------------------------------
% Original BM4D code comments:
% authors:               Matteo Maggioni
%                        Alessandro Foi
%
% web page:              http://www.cs.tut.fi/~foi/GCF-BM3D
%
% contact:               firstname.lastname@tut.fi
%
% --------------------------------------------------------------------------------------------
% Copyright (c) 2010-2015 Tampere University of Technology.
% All rights reserved.
% This work should be used for nonprofit purposes only.
% --------------------------------------------------------------------------------------------
%
% Disclaimer
% ----------
%
% Any unauthorized use of these routines for industrial or profit-oriented activities is
% expressively prohibited. By downloading and/or using any of these files, you implicitly
% agree to all the terms of the TUT limited license (included in the file Legal_Notice.txt).
% --------------------------------------------------------------------------------------------
%


%BM4D recon as a funciton
%inputs:
% kspace: the data in the NUFFT domain
% G1st: the NUFFT object for the first iteration.
% iter_nbr: number of iterations to run the recon
% data_std: standard deviation noise estimate of the volume
% alpha: determines how large standard deviation of the excitation noise is
% beta: determines how large standard deviation of the excitation noise is

%outputs:
% y_tilde_k: The magnitude output of the recon
% phi_tilde_k: The phase output of the recon


function [y_tilde_k, phi_tilde_k]  = bm4d_reconNUFFTfxn(kspace, G1st, nl, iter_nbr, data_std, alpha, beta)

% load constants
C = helper.constantsSparseTraj3D;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% modifiable parameters
data       = C.COMPLEX;      % REAL COMPLEX, default is C.COMPLEX

n          = G1st.Nx;           % size of the 3d phantom (power of 2 <=256)
data_std   = data_std/100;          % AWGN standard deviation in the initial observed data (%) (/100)


% low_pass   = 9;            % number of retained phase coefficients (per dimension)
% excursion  = 4;            % phase excursion (>=1)
min_norm   = eps;          % minimum normalized p-norm required to continue
pnorm      = 2;            % norm type used in early-stop condition

%iter_nbr   = 1e2;          % max number of iterations, default is 1e3
% alpha      = 1.010;        % alpha noise-excitation
% beta       = 5.0e2;        % beta noise-excitation

% tol        = 1.0;          % tolerance error of coverage (%)
% rot_deg    = 1*1e0;        % rotation degrees between consecutive trajectories
% line_nbr   = 1;            % number of subsampling trajectories per slice
% line_std   = 0.0;          % noise in subsampling trajectories (%)

lapse      = 10;           % lapse between saved slices during reconstruction
verbose    = C.IMAGE;        % NONE TEXT IMAGE
save_mat   = 0;            % save result to matlab .mat file

rec_type   = C.REC_3D;       % REC_2D REC_3D

do_wiener  = 1;            % perform BM4D Wiener filtering (1, 0)
profile    = 'lc';         % BM4D parameter profile ('lc', 'np', 'mp')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       MODIFY BELOW THIS POINT ONLY IF YOU KNOW WHAT YOU ARE DOING       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


size1 = G1st.Nx; size2 = G1st.Ny; size3 = G1st.Nz;

% initial seeds
randn('seed',0);
rand('seed',0);



kx1st = G1st.Kx; ky1st = G1st.Ky; kz1st = G1st.Kz;

Nz = G1st.Nz;
nx = length(kx1st)/Nz; % assumes each spiral slice is the same length, can fix this later to include different spiral setups
kx1st = reshape(kx1st, [nx Nz]);
ky1st = reshape(ky1st, [nx Nz]);
kz1st = reshape(kz1st, [nx Nz]);

% Full k-space sampling is filled up using the interleaves of the spiral
kxtmp = zeros(size(kx1st,1), size(kx1st,2), nl);
for i = 1:nl
    ktmp(:,:,i) = (kx1st + 1i*ky1st)*exp(1i*2*pi*(i-1)/nl);
end
kx = real(ktmp);
ky = imag(ktmp);
kz = repmat(kz1st,[1 1 nl]);


S = zeros(size(kx(:))); % ones where we sampling, zero where we don't
S(1:nx*Nz) = ones(1,nx*Nz); % assumes we are only sampling first interleaf, 
%and same spiral trajectory is used for each kz slice


masknufft = ones(n,n,Nz);

% For iterations
G = NUFFT(col(kx), col(ky), col(kz), n, n, Nz, 'mask', col(logical(masknufft)),'VoxelBasis','delta','InterpMethod','sparse');

theta_hat_0 = kspace; % initial k-space

%figure;plot3(kx1st(:),ky1st(:),kz1st(:));
figure;plot3(kx,ky,kz);


%figure;im(reshape(log10(abs(theta_hat_0)),[nx Nz]));axis square;colormap gray;colorbar;


% density compensation for hte NUFFT object
% 1st iteration
area_out1st = zeros(size(kx1st));
for i = 1:Nz
    kytmp = squeeze(ky1st(:,i,:));
    kxtmp = squeeze(kx1st(:,i,:));
    
    tmp= weight_vor(col(kxtmp),col(kytmp),1,0);
    tmp = reshape(tmp,[size(kxtmp,1) 1 size(kxtmp,2)]);
    area_out1st(:,i,:) = tmp;
end

% other iterations
area_out = zeros(size(kx));
for i = 1:Nz
    kytmp = squeeze(ky(:,i,:));
    kxtmp = squeeze(kx(:,i,:));
    
    tmp= weight_vor(col(kxtmp),col(kytmp),nl,0);
    tmp = reshape(tmp,[size(kxtmp,1) 1 size(kxtmp,2)]);
    area_out(:,i,:) = tmp;
end

% initial data (like zero-filled initial recon)
AAA = G1st' * (theta_hat_0(:) .* area_out1st(:))/(n*n*Nz);
if data==C.COMPLEX
    y_hat_0   = abs(AAA);
    phi_hat_0 = angle(AAA);
else
    y_hat_0   = real(AAA);
    phi_hat_0 = zeros(size(y_hat_0));
end

y_hat_0 = reshape(y_hat_0,[n n Nz]);
phi_hat_0 = reshape(phi_hat_0,[n n Nz]);

% figure;im(y_hat_0, [0 3.5e-6]);colorbar; colormap gray;
% figure;im(phi_hat_0);colorbar; colormap gray;

% figure;im(y_hat_0);colorbar; colormap gray;
% figure;im(phi_hat_0);colorbar; colormap gray;


% figure;im(reshape(log10(abs(theta_hat_0)),[nx n]));axis square;
% %figure;im(reshape(area_out,[nx n nl]));axis square;
% figure;plot3(kx(:),ky(:),kz(:));title('all of kspace'); % whole kspace being used for bm4d
% figure;plot3(kx1st,ky1st,kz1st);title('1st (0th) iteration only');

% y_hat_0 = abs(z); % perfect intial guess, what happens? NOT REAL SIMULATION
% phi_hat_0 = angle(z);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% getting sigma to be used in the first iteration
if data_std~=0
    sigma0 = data_std;
else
    sigma0 = 1000/eps;
end

% excitation noise
%min_std_excite  = 10^(-255/20)/sqrt(1-sum(S(:))/numel(y_hat_0)); % original
% edit needed for a real spiral trajectory?
min_std_excite  = 10^(-255/20)/sqrt(1-sum(S(:))/numel(kx));
min_std_excite  = (min_std_excite~=Inf) * min_std_excite;
std_excite      = max(min_std_excite,sqrt(alpha.^(-(1:iter_nbr)-beta))) + data_std;
if iter_nbr ~= 0 % to allow for 0 iteration input into function.
    std_excite(end) = 0;
end
%%%%%%%%%%%%%
% if want to get figure 7.
% figure;plot([1:iter_nbr],std_excite*100);xlabel('iteration');ylabel('\sigma (%)');
% title('Like figure 7 Maggioni et al');


% parameters' initialization
y_hat_k         = y_hat_0;
y_tilde_k       = y_hat_0;
y_hat_excite_k  = y_hat_0;
phi_hat_k       = phi_hat_0;
phi_tilde_k     = phi_hat_0;
sigma           = sigma0;
realcoverage    = sum(S(:))/numel(S)*100;
psnr_tilde      = zeros(1,iter_nbr);
psnr_hat        = zeros(1,iter_nbr);
ssim_tilde      = zeros(1,iter_nbr);
ssim_hat        = zeros(1,iter_nbr);
psnr_tilde_phi  = zeros(1,iter_nbr);
psnr_hat_phi    = zeros(1,iter_nbr);
ssim_tilde_phi  = zeros(1,iter_nbr);
ssim_hat_phi    = zeros(1,iter_nbr);
xSection        = ceil(size3/2);
%psnr_ind        = y>10/255;
sw              = [1 1 1];

buffer_cplx     = (y_hat_0.*exp(1i*phi_hat_0)) ./ sigma0^2;
weight          = 1/sigma0^2;

prev_y_tilde_k  = abs(buffer_cplx) ./ weight;

% % initializing progression variable
% if data==C.COMPLEX
%     progress        = zeros([size1 3*size2 iter_nbr/lapse+1]);
%     progress(:,:,1) = [y_hat_0(:,:,xSection) y_hat_0(:,:,xSection) ...
%         phi_hat_0(:,:,xSection)/2/pi+0.5];
% else
%     progress        = zeros([size1 2*size2 iter_nbr/lapse+1]);
%     progress(:,:,1) = [y_hat_0(:,:,xSection) y_hat_0(:,:,xSection)];
% end
idx = 2;

% % print initial information
% if verbose~=C.NONE
%     fprintf('\n%s %s phantom of size %dx%dx%d \n', C.OBS{data},C.DATA{phantom},size1,size2,size3);
%     fprintf('%s trajectory with %.2f%% subsampling \n', C.TRAJ{trajectory},realcoverage);
%     if data_std>0
%         fprintf('AWGN with sigma %.2f%% \n\n',data_std*100);
%     end
% end

% noise distribution
if data==C.COMPLEX
    distribution = 'Rice';
else
    distribution = 'Gauss';
end

start = tic;
k = 1;
early_stop = 0;
while k<=iter_nbr && ~early_stop
    % start counting
    iter = tic;
    
    if k>1
        % getting right sigma
        sigma = std_excite(k-1);
        % magnitude regularization
        y_hat_k = bm4d(y_hat_excite_k, distribution, sigma, profile, do_wiener, 0);
        
        % phase regularization
        if data==C.COMPLEX
            rand_phase_shift = 2*rand*pi-pi;
            phi_hat_k = bm4d(mod(phi_hat_k+pi+rand_phase_shift,2*pi)/2/pi, ...
                'Gauss', sigma, profile, do_wiener, 0);
            phi_hat_k = phi_hat_k*2*pi-pi-rand_phase_shift;
            phi_hat_k = mod(phi_hat_k+pi,2*pi)-pi;
        end
        
        % buffer update
        weight = weight + 1/sigma^2;
        buffer_cplx = buffer_cplx + (y_hat_k.*exp(1i*phi_hat_k))/sigma^2;
        
        if data==C.COMPLEX
            y_tilde_k   = abs(buffer_cplx./ weight);
            phi_tilde_k = angle(buffer_cplx./ weight);
        else
            y_tilde_k   = real(buffer_cplx./ weight);
            phi_tilde_k = zeros(size(y_tilde_k));
        end
        
        % early stop condition
        early_stop      = (sum(abs(prev_y_tilde_k(:)-y_tilde_k(:)).^pnorm)).^(1/pnorm) / ...
            (numel(y_tilde_k)).^(1/pnorm) < min_norm;
        prev_y_tilde_k  = y_tilde_k;
    end % moved end here, original does not include initial phase in 1st iteration, why not?  
        % excitation (prepare)
        if data==C.COMPLEX
            y_hat_excite_k = y_hat_k.*exp(1i*phi_tilde_k);
        else
            y_hat_excite_k = y_hat_k;
        end
%         if data==C.COMPLEX % why not do this?
%             y_hat_excite_k = y_tilde_k.*exp(1i*phi_tilde_k);
%         else
%             y_hat_excite_k = y_tilde_k;
%         end
    %end % where end originally was
    
    % excitation (excite)
    y_hat_excite_k    = y_hat_excite_k  + std_excite(k)*randn(size(y_hat_excite_k)) + ...
       1i*std_excite(k)*randn(size(y_hat_excite_k));
   
    theta_hat_k = G * y_hat_excite_k(:);
    
    theta_hat_k(S==1) = theta_hat_0;
    
    % for debugging - how different is theta_hat_k from the right answer in
    % kspace?
    if 0        
        figure;im(reshape(i_t_theta_hat_k,[n n Nz]));
        figure;im(reshape(y_hat_excite_k,[n n Nz]));colormap gray;colorbar;
        figure;im(reshape(y_hat_k,[n n Nz]));colormap gray;colorbar;
    end
        
    i_t_theta_hat_k   = G' * (theta_hat_k(:) .* area_out(:))/(n*n*Nz);
    i_t_theta_hat_k = reshape(i_t_theta_hat_k,[n n Nz]);

    
    y_hat_excite_k    = abs(i_t_theta_hat_k);
    if data==C.COMPLEX
        phi_hat_k = angle(i_t_theta_hat_k).*(y_hat_excite_k~=0) + ...
            (y_hat_excite_k==0).*phi_hat_k;
    end
    
    % performances
%     psnr_tilde(k)     = 10*log10(1/mean((y(psnr_ind)-y_tilde_k(psnr_ind)).^2));
%     psnr_hat(k)       = 10*log10(1/mean((y(psnr_ind)-y_hat_k(psnr_ind)).^2));
%     ssim_tilde(k)     = ssim_index3d(y_tilde_k*255,y*255,sw,psnr_ind);
%     ssim_hat(k)       = ssim_index3d(y_hat_k*255,y*255,sw,psnr_ind);
%     if data==C.COMPLEX
%         psnr_tilde_phi(k) = 10*log10(1/mean(((phi(psnr_ind)-phi_tilde_k(psnr_ind))/2/pi).^2));
%         psnr_hat_phi(k)   = 10*log10(1/mean(((phi(psnr_ind)-phi_hat_k(psnr_ind))/2/pi).^2));
%         ssim_tilde_phi(k) = ssim_index3d((phi_tilde_k/2/pi+0.5)*255,(phi/2/pi+0.5)*255,sw,psnr_ind);
%         ssim_hat_phi(k)   = ssim_index3d((phi_hat_k/2/pi+0.5)*255,(phi/2/pi+0.5)*255,sw,psnr_ind);
%     end
    
    % stop counting
    iter_time = toc(iter);
    
    % saving cross section of the progression
    if mod(k,lapse)==0
        if data==C.COMPLEX
            progress(:,:,idx) = [y_hat_excite_k(:,:,xSection) ...
                y_tilde_k(:,:,xSection) phi_tilde_k(:,:,xSection)/2/pi+0.5];
        else
            progress(:,:,idx) = [y_hat_excite_k(:,:,xSection) y_tilde_k(:,:,xSection)];
        end
        idx = idx+1;
    end
    
    % storing data to disk
    if save_mat
        if data==C.COMPLEX
            save([C.DATA{phantom},'_',C.TRAJ{trajectory},'_cov',...
                num2str(coverage),'_sigma',num2str(data_std*100),'_COMPLEX.mat'],...
                'S','y','progress','psnr_hat','psnr_tilde','ssim_tilde','ssim_hat',...
                'psnr_hat_phi','psnr_tilde_phi','ssim_tilde_phi','ssim_hat_phi',...
                'iter_nbr','alpha','beta','std_excite','data_std','y_hat_k','y_tilde_k','y_hat_0',...
                'phi','phi_hat_k','phi_hat_0','phi_tilde_k','low_pass','excursion')
        else
            save([C.DATA{phantom},'_',C.TRAJ{trajectory},'_cov',...
                num2str(coverage),'_sigma',num2str(data_std*100),'_REAL.mat'],...
                'S','y','progress','psnr_hat','psnr_tilde','ssim_tilde','ssim_hat',...
                'iter_nbr','alpha','beta','std_excite','data_std','y_hat_k','y_tilde_k','y_hat_0')
        end
    end
    
    %%% OUTPUT CODE
    if verbose==C.IMAGE
        if ~ishandle(1)
            figure(1),
        end
        if data==C.COMPLEX
            

%   CAN COMMENT THIS BACK IN comment this in if want to see iterations.
            figure(1);
            imshow([ y_hat_0(:,:,xSection) ...
                y_hat_excite_k(:,:,xSection) ...
                y_hat_k(:,:,xSection)...
                y_tilde_k(:,:,xSection)], ...
                [min(y_hat_0,[],'all') max(y_hat_0,[],'all')],...
                'InitialMagnification','fit');colorbar;
            title('y_h_a_t^0, y_h_a_t_e_x_c_i_t_e^k, y_h_a_t^k, y_t_i_l_d_e^k');
            xlabel(['k = ' num2str(k)]);
            figure(2);
            imshow([ phi_hat_0(:,:,xSection)/(2*pi)+0.5 ...
                phi_hat_k(:,:,xSection)/(2*pi)+0.5 ...
                phi_tilde_k(:,:,xSection)/(2*pi)+0.5], ...
                [min(phi_hat_0,[],'all') max(phi_hat_0,[],'all')],...
                'InitialMagnification','fit');colorbar;
            title('\phi_h_a_t^0, \phi_h_a_t^k, \phi_t_i_l_d_e^k');
            
            
            
            
%             title(sprintf(['Iteration #%04d (%.1fsec) - Excitation sigma %.2f%% \n',...
%                 'Magnitude PSNR %.2fdB SSIM %.2f - Phase PSNR %.2fdB SSIM %.2f \n',...
%                 '%s subsampling %.2f%% - Initial sigma %.2f%% \n'],...
%                 k, iter_time, sigma*100, psnr_tilde(k), ssim_tilde(k), ...
%                 psnr_tilde_phi(k), ssim_tilde_phi(k), C.TRAJ{trajectory}, ...
%                 realcoverage, data_std*100));
%             text(size2*0.5,0*size1-(1.5),'$y$','Interpreter','Latex',...
%                 'HorizontalAlignment','center','VerticalAlignment','bottom');
%             text(size2*1.5,0*size1-(1.5),'$\phi$','Interpreter','Latex',...
%                 'HorizontalAlignment','center','VerticalAlignment','bottom');
%             text(size2*2.5,0*size1-(1.5),'$\hat{y}^{(0)}$','Interpreter','Latex',...
%                 'HorizontalAlignment','center','VerticalAlignment','bottom');
%             text(size2*3.5,0*size1-(1.5),'$\hat{\phi}^{(0)}$','Interpreter','Latex',...
%                 'HorizontalAlignment','center','VerticalAlignment','bottom');
%             text(size2*0.5,2*size1+(1.5),'$\hat{y}^{(k)}$','Interpreter','Latex',...
%                 'HorizontalAlignment','center','VerticalAlignment','top');
%             text(size2*1.5,2*size1+(1.5),'$\hat{\phi}^{(k)}$','Interpreter','Latex',...
%                 'HorizontalAlignment','center','VerticalAlignment','top');
%             text(size2*2.5,2*size1+(1.5),'$\tilde{y}^{(k)}$','Interpreter','Latex',...
%                 'HorizontalAlignment','center','VerticalAlignment','top');
%             text(size2*3.5,2*size1+(1.5),'$\tilde{\phi}^{(k)}$','Interpreter','Latex',...
%                 'HorizontalAlignment','center','VerticalAlignment','top');
%             pos_caption_y = 2;
%             pos_caption_x = 2;
        else
            imshow([ y(:,:,xSection) ...
                y_hat_0(:,:,xSection) ...
                y_hat_excite_k(:,:,xSection) ...
                y_hat_k(:,:,xSection)...
                y_tilde_k(:,:,xSection) ], ...
                'InitialMagnification','fit');
%             title(sprintf(['Iteration #%04d (%.1fsec) - ',...
%                 'Excitation sigma %.2f%% - PSNR %.2fdB \n',...
%                 '%s of subsampling %.2f%% - Initial sigma %.2f%% \n'],...
%                 k, iter_time, sigma*100, psnr_tilde(k), ...
%                 TRAJ{trajectory}, realcoverage, data_std*100));
            title(sprintf(['Iteration #%04d (%.1fsec) - ',...
                'Excitation sigma %.2f%% - PSNR %.2fdB \n',...
                '%s of subsampling %.2f%% - Initial sigma %.2f%% \n'],...
                k, iter_time, sigma*100, psnr_tilde(k), ...
                realcoverage, data_std*100));
            text(size2*0.5,size1+(1.5),'$y$','Interpreter','Latex',...
                'HorizontalAlignment','center','VerticalAlignment','top');
            text(size2*1.5,size1+(1.5),'$\hat{y}^{(0)}$','Interpreter','Latex',...
                'HorizontalAlignment','center','VerticalAlignment','top');
            text(size2*2.5,size1+(1.5),'$\hat{y}_{\mathrm{excite}}^{(k)}$','Interpreter','Latex',...
                'HorizontalAlignment','center','VerticalAlignment','top');
            text(size2*3.5,size1+(1.5),'$\hat{y}^{(k)}$','Interpreter','Latex',...
                'HorizontalAlignment','center','VerticalAlignment','top');
            text(size2*4.5,size1+(1.5),'$\tilde{y}^{(k)}$','Interpreter','Latex',...
                'HorizontalAlignment','center','VerticalAlignment','top');
            pos_caption_y = 1;
            pos_caption_x = 2.5;
        end
%         text(pos_caption_x*size2,pos_caption_y*size1+4,...
%             {' ',['Phantom size ',num2str(size1),'x',num2str(size2),'x',...
%             num2str(size3),',  displaying cross-section ',num2str(xSection),' of ',num2str(size3)]},...
%             'HorizontalAlignment','center','VerticalAlignment','top');
%         pause(eps)
%         drawnow
    end
    if verbose==C.TEXT
        if data==C.COMPLEX
            fprintf('#%04d | %.2fsec - sigma %.2f%% - Magnitude %.2fdB - Phase %.2fdB \n',...
                k, iter_time, std_excite(k)*100, psnr_tilde(k), psnr_tilde_phi(k));
        else
            fprintf('#%04d | %.2fsec - sigma %.2f%% - %.2fdB \n',...
                k, iter_time, std_excite(k)*100, psnr_tilde(k));
        end
    end
    k = k + 1;
end
exec_time = toc(start);
% fprintf('Reconstruction from trajectory %s (%.2f%%) done (%.2fsec - %.2fdB) \n', ...
%     C.TRAJ{trajectory}, realcoverage, exec_time, psnr_tilde(end));

% if verbose==C.IMAGE
%     % show PSNR progression
%     figure,
%     hold all
%     plot(1:iter_nbr,psnr_tilde)
%     plot(1:iter_nbr,psnr_hat)
%     box
%     grid
%     h = legend('$\tilde{y}^{(k)}$','$\hat{y}^{(k)}$', 'Location','Best');
%     set(h,'Interpreter','Latex')
%     xlabel('Iteration','Interpreter','Latex','FontSize',11)
%     ylabel('PSNR (dB)','Interpreter','Latex','FontSize',11)
%     set(gca,'xtick',0:iter_nbr/10:iter_nbr)
%     hold off
%     
%     % compare y_tilde and y_hat PSNRs
%     figure,
%     hold all
%     plot(1:iter_nbr,psnr_tilde./psnr_hat,'-k')
%     box
%     grid
%     xlabel('Iteration','Interpreter','Latex','FontSize',11)
%     ylabel('PSNR $\tilde{y}^{(k)}$ / PSNR $\hat{y}^{(k)}$',...
%         'Interpreter','Latex','FontSize',11)
%     set(gca,'xtick',0:iter_nbr/10:iter_nbr)
%     hold off
%     
%     if data==C.COMPLEX
%         % same as before, but relative to phase reconstruction
%         figure,
%         hold all
%         plot(1:iter_nbr,psnr_tilde_phi)
%         plot(1:iter_nbr,psnr_hat_phi)
%         box
%         grid
%         h = legend('$\tilde{\phi}^{(k)}$','$\hat{\phi}^{(k)}$', 'Location','Best');
%         set(h,'Interpreter','Latex')
%         xlabel('Iteration','Interpreter','Latex','FontSize',11)
%         ylabel('PSNR (dB)','Interpreter','Latex','FontSize',11)
%         set(gca,'xtick',0:iter_nbr/10:iter_nbr)
%         hold off
%         
%         figure,
%         hold all
%         plot(1:iter_nbr,psnr_tilde_phi./psnr_hat_phi,'-k')
%         box
%         grid
%         xlabel('Iteration','Interpreter','Latex','FontSize',11)
%         ylabel('PSNR $\tilde{\phi}^{(k)}$ / PSNR $\hat{\phi}^{(k)}$',...
%             'Interpreter','Latex','FontSize',11)
%         set(gca,'xtick',0:iter_nbr/10:iter_nbr)
%         hold off
%     end
%     
%     % show 3-D cross-sections
%     if data==C.COMPLEX
%         volumes = {y,phi/2/pi+0.5,y_tilde_k,phi_tilde_k/2/pi+0.5};
%         fig_title = {'Original magnitude','Original phase',...
%             'Magnitude estimate','Phase estimate'};
%         visualizeXsect( volumes, fig_title );
%     else
%         volumes = {y,y_tilde_k};
%         fig_title = {'Original phantom','Final estimate'};
%         visualizeXsect( volumes, fig_title );
%     end
% end

% figure;im(y_tilde_k);colormap gray;colorbar;
% figure;im(phi_tilde_k);colormap gray;colorbar;


end
