
% code to reconstruct a digital phantom, an adaptation from the original
% BM4D LEGACY RELEASE github dem_reconstruction code from the LEGACY RELEASED from https://webpages.tuni.fi/foi/GCF-BM3D/ 


restoredefaultpath

addpath(genpath('/shared/mrfil-data/cmm15/cmarchinimatlab/BM4DNUFFT2'));

% add the path to the original BM4D code from (BM4D_v3p2), get it at
% https://webpages.tuni.fi/foi/GCF-BM3D/
addpath(genpath('/shared/mrfil-data/cmm15/Algorithms/BM4D_v3p2/'));


close all;clear;
%% get phantom
n = 64; Nz = 16;
low_pass   = 9;            % number of retained phase coefficients (per dimension)
excursion  = 4;            % phase excursion (>=1)
data_std   = 5.0;          % AWGN standard deviation in the initial observed data (%)
iterations = 0;
Nx = n; Ny = n;
nSlices = Nz; Nslices = Nz;

alpha      = 1.010;        % alpha noise-excitation
beta       = 5.0e2;        % beta noise-excitation

% initial seeds
randn('seed',0);
rand('seed',0);

% padding phantom
brainweb = 't1_icbm_normal_1mm_pn0_rf0.rawb';
if ~exist(brainweb,'file')
    error(['Could not read BrainWeb phantom "',brainweb,'" file. ',...
        'Please download it from http://mouldy.bic.mni.mcgill.ca/'...
        'brainweb/selection_normal.html ',...
        '(Modality T1, Slice thickness 1mm, ',...
        'Noise 0%, Intensity non-uniformity 0%)']);
end
% load BrainWeb phantom and scale it to size n
fid = fopen(brainweb);
y = double(reshape(fread(fid,181*217*181),[181 217 181]));
fclose(fid);
w = zeros([217,217,217]);
w(19:181+18,:,19:181+18) = y;
% scale x-y
s = n/217;
w = imresize(w,s,'bilinear');
% scale z
y = zeros(n,n,n);
for i=1:n
    for j=1:n
        y(i,j,:) = imresize(squeeze(w(i,j,:)),s,'bilinear');
    end
end
clear w s
% make Nz slices thick instead of n
y = y(:,:,n/2-Nz/2+1:n/2+Nz/2);
y = (y-min(y(:))) / (max(y(:))-min(y(:)));
phi = 2*pi*0.25*randn(size(y));
mask = zeros(size(phi));
% if rec_type==C.REC_2D
%     mask(1:low_pass,1:low_pass) = 1;
% else
mask(1:low_pass,1:low_pass,1:low_pass) = 1;
% end
phi = helper.idct3(helper.dct3(phi).*mask);
R = max(phi(:))-min(phi(:));
phi = mod(2*pi*phi/R*max(1,excursion)+pi,2*pi)-pi;
z = y.*exp(1i*phi);
data_std = data_std/100;
z = z + data_std/sqrt(2)*(randn(size(y)) + 1i*randn(size(y)));
data_std = data_std*100; % to input into bm4d again.

% figure;im(abs(z));colormap gray; colorbar;
% figure;im(angle(z));colormap gray; colorbar;

%%
% getting a spiral kspace trajectory:
nl = 3; % number of interleaves
alphavd = 6; % determines how sampling density changes
D = 24;  % 
gamp = 3;
gslew = 120;
gts = 3e-6; % gradient raster time in sec
[~, ~, kx, ky, ~, ~] = genspivd_Kim(D, Nx, nl, gamp, gslew, gts, alphavd);



kx = repmat(kx.',[1 nSlices]); ky = repmat(ky.', [1 nSlices]);
%kz = repmat([-8:7], [size(kx,1) 1]);
kz = repmat([-7:8], [size(kx,1) 1]);



figure;plot3(kx(:),ky(:),kz(:)); % k-space trajectory for acquired data

% the NUFFT object for the acquired data
mask = ones(Nx,Ny,Nslices);
G = NUFFT(col(kx), col(ky), col(kz), Nx, Ny, Nslices, 'mask', col(logical(mask)),'VoxelBasis','delta','InterpMethod','sparse');

Np = size(kx,1);

% simulate initial kspace acquisition.
kspace = G * z(:);
kspace = reshape(kspace,[Np Nz]);

% % coil combination:
% kspacetmp = reshape(kspace,[size(kspace,1)*size(kspace,2)*size(kspace,3) size(kspace,4)]);
% [~,~,V] = svd(kspacetmp,0);
% kspace = reshape(kspacetmp*V(:,1), [size(kspace,1) size(kspace,2) size(kspace,3)]);

figure;im(squeeze((log10(abs(kspace(:,:,1))))));colorbar;axis square;

%img = zeros(Nx,Ny,Nslices,size(kspace,3));
for i = 1:size(kspace,3)
    % run the BM4D reconstruction for each time point (only 1 is simulated
    % here)
    [magnitude(:,:,:,i), phase(:,:,:,i)] = bm4d_reconNUFFTfxn(col(kspace(:,:,i)), G, nl, iterations, data_std, alpha, beta);
end

img = magnitude.*exp(1i*phase);


r = [0 max(abs(img),[],'all')];
%r = [0 5e-9];
figure;im(abs(img(:,:,:,1)),r);xlabel('recon');
colormap gray; colorbar;

% figure;im(abs(img(:,:,:,2)),r);xlabel('tag');
% colormap gray;
% 
% figure;im(mean(abs(img(:,:,:,1:2:end)) - abs(img(:,:,:,2:2:end)),4));
% xlabel('mean perfusion');
% colormap gray; colorbar;

%% look at metrics

ssimvalmag = ssim(abs(img), abs(y));
ssimvalphase = ssim(angle(img), angle(y));

%psnr();




