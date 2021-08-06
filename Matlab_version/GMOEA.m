function GMOEA(Global)
% <algorithm> <S>
% Evolutionary Multi-Objective Optimization Driven by Generative Adversarial Networks (GANs)

%------------------------------- Reference --------------------------------

%------------------------------- Copyright --------------------------------
% Copyright (c) 2018-2019 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    [wmax,k,lr,batchsize,epoch] = Global.ParameterSet(5,10,0.001,16,200);
    %% Generate random population
    Population = Global.Initialization(2*Global.N);
    %% Build GAN
    %pyversion D:\ProgramData\Anaconda3\python.exe
    import GAN_model.* 
    net = py.GAN(Global.D,batchsize,lr,epoch,Global.D);
    while Global.NotTermination(Population)
        [~,index,~] = EnvironmentalSelection(Population,k);
        [Input,Labels,Pool] = Precessing(Global,Population,index);
        net         = net.train(Input,Labels,Pool);

        for i = 1 : wmax
            Offspring = NetGenerate(Global,net,Population(index).decs);
        end
        [Population,~,Fitness] = EnvironmentalSelection([Population,Offspring],Global.N);
    end
end


function [Input,Label,pool] = Precessing(Global,Population,index)
    Label = zeros(2*Global.N,1);
    Label(index) = 1;
    pool  =  Population(index).decs ./ repmat(Global.upper,k,1);
    Input = (Population.decs-repmat(Global.lower,2*Global.N,1)) ./ ...
            repmat(Global.upper-Global.lower,2*Global.N,1);
end

function PopDec = NetGenerate(Global,net,PopDec)
    OffDec = net.generate(PopDec./ repmat(Global.upper, size(PopDec,1),1), Global.N) .* ...
            repmat(Global.upper, Global.N, 1);
    Offspring = PM(Global, OffDec);
end

function Offspring = PM(Global, PopDec)
    [proC,disC,proM,disM] = deal(1,20,1,20);
    Parent1 = PopDec(1:floor(end/2),:);
    Parent2 = PopDec(floor(end/2)+1:floor(end/2)*2,:);
    [N,D] = size(PopDec);
    beta = zeros(N,D);
    mu   = rand(N,D);
    beta(mu<=0.5) = (2*mu(mu<=0.5)).^(1/(disC+1));
    beta(mu>0.5)  = (2-2*mu(mu>0.5)).^(-1/(disC+1));
    beta = beta.*(-1).^randi([0,1],N,D);
    beta(rand(N,D)<0.5) = 1;
    beta(repmat(rand(N,1)>proC,1,D)) = 1;
    Offspring = [(Parent1+Parent2)/2+beta.*(Parent1-Parent2)/2
                 (Parent1+Parent2)/2-beta.*(Parent1-Parent2)/2];
    % Polynomial mutation
    Lower = repmat(Global.lower,2*N,1);
    Upper = repmat(Global.upper,2*N,1);
    Site  = rand(2*N,D) < proM/D;
    mu    = rand(2*N,D);
    temp  = Site & mu<=0.5;
    Offspring       = min(max(Offspring,Lower),Upper);
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                      (1-(Offspring(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
    temp = Site & mu>0.5; 
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                      (1-(Upper(temp)-Offspring(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
    Offspring = INDIVIDUAL(Offspring);
end