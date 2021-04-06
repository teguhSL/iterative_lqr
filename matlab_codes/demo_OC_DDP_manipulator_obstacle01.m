function demo_OC_DDP_manipulator_obstacle01
% iLQR applied to a 2D manipulator task with obstacles avoidance
% (viapoints task with position+orientation including bounding boxes on f(x))
%
% Sylvain Calinon, 2021

addpath('./m_fcts/');


%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model.dt = 1E-2; %Time step size
model.nbData = 50; %Number of datapoints
model.nbIter = 50; %Number of iterations for iLQR
model.nbPoints = 1; %Number of viapoints
model.nbObstacles = 2; %Number of obstacles
model.nbVarX = 3; %State space dimension (q1,q2,q3)
model.nbVarU = 3; %Control space dimension (dq1,dq2,dq3)
model.nbVarF = 3; %Objective function dimension (x1,x2,o)
model.l = [2, 2, 1]; %Robot links lengths
model.sz = [.2, .3]; %Size of objects
model.sz2 = [.5, .8]; %Size of obstacles
model.q = 1E2; %Tracking weight term
model.q2 = 1E0; %Obstacle avoidance weight term
model.r = 1E-3; %Control weight term

model.Mu = [3; -1; 0]; %Viapoints (x1,x2,o)
for t=1:model.nbPoints
	model.A(:,:,t) = [cos(model.Mu(3,t)), -sin(model.Mu(3,t)); sin(model.Mu(3,t)), cos(model.Mu(3,t))]; %Orientation
end

model.Mu2 = [[2.5; 2.0; pi/4], [3.5; .5; -pi/6]]; %Obstacles (x1,x2,o)
% model.Mu2 = [[-1.2; 1.5; pi/4], [-0.5; 2.5; -pi/6]]; %Obstacles (x1,x2,o)
for t=1:model.nbObstacles
	model.A2(:,:,t) = [cos(model.Mu2(3,t)), -sin(model.Mu2(3,t)); sin(model.Mu2(3,t)), cos(model.Mu2(3,t))]; %Orientation
	model.S2(:,:,t) = model.A2(:,:,t) * diag(model.sz2).^2 * model.A2(:,:,t)'; %Covariance matrix
end
model.th = -0.5 - 0.1; %Threshold to describe obstacle boundary (quadratic form)
% model.th = exp(-0.5 - 0.1); %Threshold to describe obstacle boundary (exponential form)

Q = speye(model.nbVarX * model.nbPoints) * model.q; %Precision matrix to reach viapoints
% Q = kron(eye(model.nbPoints), diag([1E0, 1E0, 0])); %Precision matrix (by removing orientation constraint)
R = speye((model.nbData-1) * model.nbVarU) * model.r; %Control weight matrix (at trajectory level)

%Time occurrence of viapoints
tl = linspace(1, model.nbData, model.nbPoints+1);
tl = round(tl(2:end));
idx = (tl - 1) * model.nbVarX + [1:model.nbVarX]';


%% Iterative LQR (iLQR)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
u = zeros(model.nbVarU*(model.nbData-1), 1); %Initial commands
% u = [ones(model.nbData-1, 1)*1E0; zeros(2*(model.nbData-1), 1)]; %Initial commands
x0 = [3*pi/4; -pi/2; -pi/4]; %Initial robot pose

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
	cost0 = f(:)' * Q * f(:) + norm(f2(:))^2 * model.q2 + norm(u)^2 * model.r; %u' * R * u
	while 1
		utmp = u + du * alpha;
		xtmp = reshape(Su0 * utmp + Sx0 * x0, model.nbVarX, model.nbData);
		ftmp = f_reach(xtmp(:,tl), model); %Tracking objective
		ftmp2 = f_avoid(xtmp, model); %Avoidance objective
		cost = ftmp(:)' * Q * ftmp(:) + norm(ftmp2(:))^2 * model.q2 + norm(utmp)^2 * model.r; %utmp' * R * utmp
		if cost < cost0 || alpha < 1E-3
			break;
		end
		alpha = alpha * 0.5;
	end
	u = u + du * alpha;
	
	if norm(du * alpha) < 1E-2
		break; %Stop iLQR when solution is reached
	end
end
disp(['iLQR converged in ' num2str(n) ' iterations.']);

%Log data
r.x = x;
r.f = [];
for j=model.nbVarU:-1:1
	r.f = [r.f; fkine0(x(1:j,:), model.l(1:j))];
end
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
plotArm(r.x(:,1), model.l, [0; 0; -3], .2, [.8 .8 .8]);
plotArm(r.x(:,tl(1)), model.l, [0; 0; -2], .2, [.6 .6 .6]);
% plotArm(r.x(:,tl(2)), model.l, [0; 0; -1], .2, [.4 .4 .4]);
for t=1:model.nbPoints
	patch(msh(1,:,t), msh(2,:,t), min(colMat(t,:)*1.7,1),'linewidth',2,'edgecolor',colMat(t,:)); %,'facealpha',.2
	plot2Dframe(model.A(:,:,t)*4E-1, model.Mu(1:2,t), repmat(colMat(t,:),3,1), 6);
end
for t=1:model.nbObstacles
	plotGMM(model.Mu2(1:2,t), model.S2(:,:,t), min(colMat2(t,:)*1.7,1));
	plot2Dframe(model.A2(:,:,t)*4E-1, model.Mu2(1:2,t), repmat(colMat2(t,:),3,1), 6);
end
for j=model.nbVarU:-1:1
	plot(r.f((j-1)*model.nbVarF+1,:), r.f((j-1)*model.nbVarF+2,:), '-','linewidth',1,'color',[.2 .2 .2]);
	plot(r.f((j-1)*model.nbVarF+1,:), r.f((j-1)*model.nbVarF+2,:), '.','markersize',10,'color',[.2 .2 .2]);
end
plot(r.f(1,:), r.f(2,:), '-','linewidth',2,'color',[0 0 0]);
plot(r.f(1,:), r.f(2,:), '.','markersize',10,'color',[0 0 0]);
plot(r.f(1,[1,tl]), r.f(2,[1,tl]), '.','markersize',30,'color',[0 0 0]);
axis equal; 
% print('-dpng','graphs/DDP_manipulator_obstacle01.png');


% %% Timeline plot
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure('position',[820 10 800 800],'color',[1 1 1]); 
% %States plot
% for j=1:model.nbVarF
% 	subplot(model.nbVarF+model.nbVarU, 1, j); hold on;
% 	plot(tl, model.Mu(j,:), '.','markersize',35,'color',[.8 0 0]);
% 	plot(r.f(j,:), '-','linewidth',2,'color',[0 0 0]);
% 	ylabel(['$f_' num2str(j) '$'], 'interpreter','latex','fontsize',26);
% end
% %Commands plot
% for j=1:model.nbVarU
% 	subplot(model.nbVarF+model.nbVarU, 1, model.nbVarF+j); hold on;
% 	plot(r.u(j,:), '-','linewidth',2,'color',[0 0 0]);
% 	ylabel(['$u_' num2str(j) '$'], 'interpreter','latex','fontsize',26);
% end
% xlabel('$t$','interpreter','latex','fontsize',26); 

pause;
close all;
end 


%%%%%%%%%%%%%%%%%%%%%%
%Logarithmic map for R^2 x S^1 manifold
function e = logmap(f, f0)
	e(1:2,:) = f(1:2,:) - f0(1:2,:);
	e(3,:) = imag(log(exp(f0(3,:)*1i)' .* exp(f(3,:)*1i).'));
end

%%%%%%%%%%%%%%%%%%%%%%
%Forward kinematics (in robot coordinate system)
function f = fkine0(x, L)
	T = tril(ones(length(L)));
	f = [L * cos(T * x); ...
			 L * sin(T * x); ...
			 mod(sum(x,1)+pi, 2*pi) - pi]; %x1,x2,o (orientation as single Euler angle for planar robot)
end

%%%%%%%%%%%%%%%%%%%%%%
%Jacobian with analytical computation (for single time step)
function J = jacob0(x, L)
	T = tril(ones(length(L)));
	J = [-sin(T * x)' * T * diag(L); ...
				cos(T * x)' * T * diag(L); ...
				ones(1, length(L))]; %x1,x2,o
end

%%%%%%%%%%%%%%%%%%%%%%
%Cost and gradient for a viapoints reaching task (in object coordinate system)
function [f, J] = f_reach(x, model)
% 	f = fkine0(x, model) - model.Mu; %Error by ignoring manifold
	f = logmap(fkine0(x, model.l), model.Mu); %Error by considering manifold
	
	J = []; 
	for t=1:size(x,2)
% 		f(:,t) = logmap(fkine0(x(:,t), model.l), model.Mu(:,t));
		f(1:2,t) = model.A(:,:,t)' * f(1:2,t); %Object-centered FK
		
		Jtmp = jacob0(x(:,t), model.l);
		Jtmp(1:2,:) = model.A(:,:,t)' * Jtmp(1:2,:); %Object-centered Jacobian
		
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
	f = [];
	J = [];
	id = [];
	for i=1:model.nbObstacles
		for j=1:model.nbVarU
			for t=1:model.nbData
				xee = fkine0(x(1:j,t), model.l(1:j));
				e = xee(1:2) - model.Mu2(1:2,i);
				ftmp = -0.5 * e' / model.S2(:,:,i) * e; %quadratic form
% 			ftmp = exp(-0.5 * e' / model.S2(:,:,i) * e); %exponential form
				%Bounding boxes 
				if ftmp > model.th
					f = [f; ftmp - model.th];
					Jrob = [jacob0(x(1:j,t), model.l(1:j)), zeros(model.nbVarF, model.nbVarU-j)];
					Jtmp = (-e' / model.S2(:,:,i)) * Jrob(1:2,:); %quadratic form
% 				Jtmp = exp(-0.5 * e' / model.S2(:,:,i) * e) * (-e' / model.S2(:,:,i)); %exponential form
					J = blkdiag(J, Jtmp);
					id = [id, (t-1) * model.nbVarU + [1:model.nbVarU]'];
				end
			end
		end
	end
end