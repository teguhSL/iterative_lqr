function demo_OC_DDP_obstacle01
% iLQR applied to a 2D point-mass system reaching a target while avoiding obstacles defined by Gaussians
%
% Sylvain Calinon, 2021

addpath('./m_fcts/');


%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model.dt = 1E-2; %Time step size
model.nbData = 100; %Number of datapoints
model.nbIter = 300; %Maximum number of iterations for iLQR
model.nbPoints = 1; %Number of viapoints
model.nbObstacles = 2; %Number of obstacles
model.nbVarX = 2; %State space dimension (x1,x2)
model.nbVarU = 2; %Control space dimension (dx1,dx2)
model.sz = [.2, .2]; %Size of objects
model.sz2 = [.4, .6] * 1E0; %Size of obstacles
model.q = 1E2; %Tracking weight term
model.q2 = 1E0; %Obstacle avoidance weight term
model.r = 1E-3; %Control weight term

model.Mu = [3; 3; 0]; %Viapoints (x1,x2,o)
for t=1:model.nbPoints
	model.A(:,:,t) = [cos(model.Mu(3,t)), -sin(model.Mu(3,t)); sin(model.Mu(3,t)), cos(model.Mu(3,t))]; %Orientation
end

% model.Mu2 = [rand(2,model.nbObstacles)*3; rand(1,model.nbObstacles)*pi];
model.Mu2 = [[1; 0.6; pi/4], [2.0; 2.5; -pi/6]]; %Obstacles (x1,x2,o)
for t=1:model.nbObstacles
	model.A2(:,:,t) = [cos(model.Mu2(3,t)), -sin(model.Mu2(3,t)); sin(model.Mu2(3,t)), cos(model.Mu2(3,t))]; %Orientation
	model.S2(:,:,t) = model.A2(:,:,t) * diag(model.sz2).^2 * model.A2(:,:,t)'; %Covariance matrix
end
model.th = -0.5 - 0.1; %Threshold to describe obstacle boundary (quadratic form)
% model.th = exp(-0.5 - 0.1); %Threshold to describe obstacle boundary (exponential form)

Q = speye(model.nbVarX * model.nbPoints) * model.q; %Precision matrix to reach viapoints
R = speye((model.nbData-1) * model.nbVarU) * model.r; %Control weight matrix (at trajectory level)

%Time occurrence of viapoints
tl = linspace(1, model.nbData, model.nbPoints+1);
tl = round(tl(2:end));
idx = (tl - 1) * model.nbVarX + [1:model.nbVarX]';


%% Iterative LQR (iLQR)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
u = zeros(model.nbVarU*(model.nbData-1), 1);
x0 = zeros(model.nbVarX, 1);

%Transfer matrices (for linear system as single integrator)
Su0 = [zeros(model.nbVarX, model.nbVarX*(model.nbData-1)); tril(kron(ones(model.nbData-1), eye(model.nbVarX)*model.dt))];
Sx0 = kron(ones(model.nbData,1), eye(model.nbVarX));
Su = Su0(idx,:);

for n=1:model.nbIter
	x = reshape(Su0 * u + Sx0 * x0, model.nbVarX, model.nbData); %System evolution
	[f, J] = f_reach(x(:,tl), model); %Tracking objective
	
	[f2, J2, id2] = f_avoid(x, model); %Avoidance objective
	Su2 = Su0(id2,:);
	
	du = (Su' * J' * Q * J * Su + Su2' * J2' * J2 * Su2 * model.q2 + R) \ (-Su' * J' * Q * f(:) - Su2' * J2' * f2(:) * model.q2 - u * model.r); %Gradient
	
	%Estimate step size with backtracking line search method
	alpha = 1;
	cost0 = norm(f(:))^2 * model.q + norm(f2(:))^2 * model.q2 + norm(u)^2 * model.r;
	while 1
		utmp = u + du * alpha;
		xtmp = reshape(Su0 * utmp + Sx0 * x0, model.nbVarX, model.nbData);
		ftmp = f_reach(xtmp(:,tl), model); %Tracking objective
		ftmp2 = f_avoid(xtmp, model); %Avoidance objective
		cost = norm(ftmp(:))^2 * model.q + norm(ftmp2(:))^2 * model.q2 + norm(utmp)^2 * model.r;
		if cost < cost0 || alpha < 1E-3
			break;
		end
		alpha = alpha * 0.5;
	end
	u = u + du * alpha;
	
	if norm(du * alpha) < 5E-2
		break; %Stop iLQR when solution is reached
	end
end
disp(['iLQR converged in ' num2str(n) ' iterations.']);

%Log data
r.x = x;
r.u = reshape(u, model.nbVarU, model.nbData-1);


%% Plot state space
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
colMat = lines(model.nbPoints); %Colors for viapoints
colMat2 = repmat([.4, .4, .4], model.nbObstacles, 1); %Colors for obstacles
msh0 = diag(model.sz) * [-1 -1 1 1 -1; -1 1 1 -1 -1];
for t=1:model.nbPoints
	msh(:,:,t) = model.A(:,:,t) * msh0 + repmat(model.Mu(1:2,t), 1, size(msh0,2));
end

figure('position',[10,10,800,800],'color',[1,1,1]); hold on; axis off;
for t=1:model.nbPoints
	patch(msh(1,:,t), msh(2,:,t), min(colMat(t,:)*1.7,1),'linewidth',2,'edgecolor',colMat(t,:)); %,'facealpha',.2
	plot2Dframe(model.A(:,:,t)*4E-1, model.Mu(1:2,t), repmat(colMat(t,:),3,1), 6);
end
for t=1:model.nbObstacles
	plotGMM(model.Mu2(1:2,t), model.S2(:,:,t), min(colMat2(t,:)*1.7,1));
	plot2Dframe(model.A2(:,:,t)*diag(model.sz2)*.9, model.Mu2(1:2,t), repmat(colMat2(t,:),3,1), 2);
end
plot(r.x(1,:), r.x(2,:), '-','linewidth',2,'color',[0 0 0]);
plot(r.x(1,:), r.x(2,:), '.','markersize',10,'color',[0 0 0]);
plot(r.x(1,[1,tl]), r.x(2,[1,tl]), '.','markersize',30,'color',[0 0 0]);
axis equal; 
% print('-dpng','graphs/DDP_obstacle01.png');


% %% Timeline plot
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure('position',[820 10 800 800],'color',[1 1 1]); 
% %States plot
% for j=1:model.nbVarX
% 	subplot(model.nbVarX+model.nbVarU, 1, j); hold on;
% 	plot(tl, model.Mu(j,:), '.','markersize',35,'color',[.8 0 0]);
% 	plot(r.x(j,:), '-','linewidth',2,'color',[0 0 0]);
% 	ylabel(['$x_' num2str(j) '$'], 'interpreter','latex','fontsize',26);
% end
% %Commands plot
% for j=1:model.nbVarU
% 	subplot(model.nbVarX+model.nbVarU, 1, model.nbVarX+j); hold on;
% 	plot(r.u(j,:), '-','linewidth',2,'color',[0 0 0]);
% 	ylabel(['$u_' num2str(j) '$'], 'interpreter','latex','fontsize',26);
% end
% xlabel('$t$','interpreter','latex','fontsize',26); 

pause;
close all;
end 


%%%%%%%%%%%%%%%%%%%%%%
%Cost and gradient for a viapoints reaching task (in object coordinate system)
function [f, J] = f_reach(x, model)
	f = x - model.Mu(1:2,:); %Error by ignoring manifold
	J = [];
	for t=1:size(x,2)
		f(1:2,t) = model.A(:,:,t)' * f(1:2,t); %Object-centered FK
		Jtmp = model.A(:,:,t)'; %Object-centered Jacobian
		
% 		%Bounding boxes (optional)
% 		for i=1:2
% 			if abs(f(i,t)) < model.sz(i)
% 				f(i,t) = 0;
% 				Jtmp(i,:) = 0;
% 			else
% 				f(i,t) = f(i,t) - sign(f(i,t)) * model.sz(i);
% 			end
% 		end
		
		J = blkdiag(J, Jtmp);
	end
end

%%%%%%%%%%%%%%%%%%%%%%
%Cost and gradient for an obstacle avoidance task
function [f, J, id] = f_avoid(x, model)
	f=[]; J=[]; id=[];
	for i=1:model.nbObstacles
		for t=1:model.nbData
			e = x(:,t) - model.Mu2(1:2,i);
			ftmp = -0.5 * e' / model.S2(:,:,i) * e; %quadratic form
% 			ftmp = exp(-0.5 * e' / model.S2(:,:,i) * e); %exponential form
			%Bounding boxes 
			if ftmp > model.th
				f = [f; ftmp - model.th];
				Jtmp = -e' / model.S2(:,:,i); %quadratic form
% 				Jtmp = exp(-0.5 * e' / model.S2(:,:,i) * e) * (-e' / model.S2(:,:,i)); %exponential form
				J = blkdiag(J, Jtmp);
				id = [id, (t-1) * model.nbVarU + [1:model.nbVarU]'];
			end
		end
	end
end