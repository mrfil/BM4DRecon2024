function [Gx, Gy, kx, ky, sx, sy]=genspivd_Kim(D, N, nl, gamp, gslew, gts, alphavd); 
%function [Gx, Gy, kx, ky, sx, sy]=genspivd_Kim(D, N, nl, gamp, gslew, gts, alphavd);
%   multi- shot spiral design 
%    uses Duyn's approximate slewrate limited design 
%    augmented with archimedian gmax limit 
%    inputs (args) 
%        D = FOV, cm 
%        N = matrix size
%	 %Tmax = longest acquisition allowed, s 
%	 gts = output sample spacing, s, gradient raster time 
%        gtype = trajectory type 
%    inputs (CVs) 
%        nl = number of interleaves 
%        gamp = design grad max, G/cm 
%        gslew = design slew rate, mT/m/ms 
%	 nramp = number of rampdown points 
%    outputs 
%        Gx, Gy 
%        grev 
%    time is in sec 
% 
%		rev 0 12/26/98	original 
%		rev 1 4/15/99	little better calc of ts 
% 
%               borrowed from Doug Noll, Univ. of Michigan
%               modified to take more input cv's 
% IMplementation described in Glover, Simple Analytic spiral k-space
% algorithm  MRM 42:412-415.
  
%%%%%%%%%% Predefined variables

GRESMAX= 21000;  % Value from GE, Siemens value??
if ~exist('nl','var')
    nl=1   % Number of interleaves
end
if ~exist('gamp','var')
      gamp=4; %2.2 for GE
      		% 4 for Allegra
end
if ~exist('gslew','var')
      gslew=400; % 180 for GE
      		% 400 for Allegra
end
if ~exist('gts','var')
  gts = 10e-06;   % 4e-06 for GE
  		 %10e-06 for Siemens
end
if ~exist('alphavd','var')
  alphavd = 4;
end
  nramp=0;
% nramp=100;
MAX_PG_WAMP=32766;

Tmax = GRESMAX*gts;

dts = gts;
opfov = D

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

gamma = 4.257e3; 
gambar = gamma;
 
   gx=zeros(1,2*GRESMAX);
   gy=zeros(1,2*GRESMAX); 
 
 
  q = 5;
  S0 = gslew*100; 
  rdt = dts*.5; 
  odt = dts;
  G0 = gamp;
  
  km = .5*N/D;            % max desired k space radius	  
  res = D/N;

  t = [0:rdt:.1];

  % omega : 2*pi*n 
  % n : # of turns (when k(1)-k((n-1)/n) = 1/fov)
  % chi : when tau=1, such that res is satisfied (k(1) = km) 
  %	(thus, chi = km, since n is integer)
  % 0 <= tau <= 1 (tau is function of t) 

  % find chi and n
  chi = km;
  n = (floor(1/(1-(1-2/(N/nl))^(1/alphavd))));
  omega = 2*pi*n;
	  

  Te = sqrt(S0*gamma/(chi*omega^2));
  Tr = (gamma*G0)/(chi*omega);

  Ts=1/((alphavd/2+1)*Te);
  T1=(alphavd/2+1)*Te;
  ts=(Tr*(alphavd/2+1)/T1^((alphavd+1)/(alphavd*.5+1)))^(1+2/alphavd);
	  
  
if(ts<Ts)
  t = [0: rdt: ts];
  len = length(t);
  tau1 = ((alphavd/2+1)*sqrt((S0*gamma)/(chi*omega^2))*t).^(1/(alphavd/2+1));
  ts = floor(ts/gts)*gts;	% due to quantization error of t 	
  tm = [Te*(alphavd/2+1)*ts]^((alphavd+1)/(alphavd*.5+1))/(Tr*(alphavd+1))
  Ts = 1/(Tr*(alphavd+1))
  t = [tm+rdt: rdt: Ts];
  tau2 = ((gamma*G0)/(chi*omega)*(alphavd+1).*t).^(1/(alphavd+1));
  tau = [tau1, tau2]; 
else
  t = [0: rdt: Ts];
  tau = ((alphavd/2+1)*sqrt((S0*gamma)/(chi*omega^2))*t).^(1/(alphavd/2+1));
  len = length(t);
end

T = Ts;

k = chi * tau.^alphavd .* exp(sqrt(-1)*omega.*tau);
gx = diff(k)/gamma/rdt;

l2 = length(gx);
Gx = real(gx(1:2:l2));   % or gx(1:2:l2)*MAX_PG_WAMP/gamp
Gy = imag(gx(1:2:l2));   % or gy(1:2:l2)*MAX_PG_WAMP/gamp
g = Gx + i.*Gy;   %slew rate vector
s = diff(g)./(gts*1000);  % grad vector
Kx = cumsum([0 Gx])*gts*opfov*gambar;
Ky = cumsum([0 Gy])*gts*opfov*gambar;
k = Kx + i.*Ky;  %kspace vector
t = [0:gts:T]; % time vector
matrix = max(abs(k))*2
maxg = max(abs(g))
maxs = max(abs(s))
maxt = max(t).*1000

kx = real(k);
ky = imag(k);
sx = real(s);
sy = imag(s);
