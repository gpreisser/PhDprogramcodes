begin
	using Random
	using Statistics
	using ITensors
	using LinearAlgebra
end

begin
	using DelimitedFiles
	using PyPlot
	using Colors
	using ColorSchemes
end

function ingredients(path::String)
	# this is from the Julia source code (evalfile in base/loading.jl)
	# but with the modification that it returns the module instead of the last object
	name = Symbol(basename(path))
	m = Module(name)
	Core.eval(m,
        Expr(:toplevel,
             :(eval(x) = $(Expr(:core, :eval))($name, x)),
             :(include(x) = $(Expr(:top, :include))($name, x)),
             :(include(mapexpr::Function, x) = $(Expr(:top, :include))(mapexpr, $name, x)),
             :(include($path))))
	m
end

begin
smps = ingredients("./s3mps_v2pluto.jl");
end;

begin
N = 4;
V = 1.0;
gamn = 0;
#gamzs = [0.25;0.5;1.0;2.0;]
gamu = 4.0;
gamd = gamu;
#dt = 0.1;
dt = 0.01;
gamz = 0.0
Nt = 800;
Nts = [1:1:Nt;];
#seeds = [1:1:Nt;];
chi = 512;
tf = 10.0;
J = 1;
onejumpop = false

end;

begin
sm = [0 1; 0 0];
sz = sm'*sm - sm*sm';
sx = sm + sm';
sy = -1im*(sm - sm')
lm = sqrt(gamd)*sm
lp = sqrt(gamu)*sm'
lz = sqrt(gamz)*sz
id2 = Matrix(I,2,2);
end;

begin
	ise =  [1; 0];
	iso =  [0; 1];
	isx = (1/sqrt(2))*[1;1]
	is0 = Vector{Vector{Float64}}()
	for ll = 1:N
		if isodd(ll)
			push!(is0,iso)
		else
			push!(is0,ise)
		end
	end
end

begin

for seed in Nts
	
steps = Int(round(tf/dt)) + 1
ofn = "./data/qtmps2s_N$(N)_gamu$(gamu)_gamd$(gamd)_gamz$(gamz)_dt$(dt)_tf$(tf)_chi$(chi)_seed$(seed)";
rm(ofn, force=true)

rng = MersenneTwister(seed);
rng2 = MersenneTwister(seed+Nt);
eslength = length([0.0:1/(N*gamd):tf+1;])*20
vecrand = rand!(rng, zeros(eslength));
vecrand2 = rand!(rng2, zeros(eslength));

Hnhi = -1im*0.5*gamd*kron(sm'*sm,id2) -1im*0.5*gamu*kron(sm*sm',id2);;
Hnhm = -1im*0.25*gamd*kron(sm'*sm,id2) - 1im*0.25*gamu*kron(id2,sm*sm') -1im*0.25*gamd*kron(sm'*sm,id2) - 1im*0.25*gamu*kron(id2,sm*sm')
Hnhf = -1im*0.5*gamd*kron(id2,sm'*sm) -1im*0.5*gamu*kron(id2,sm*sm')
H_h = -(1/4)*J*(kron(sx,sx) + kron(sy, sy)) + (V/4)*kron(sz,sz);
H_i = H_h + Hnhi
H_m = H_h + Hnhm
H_f = H_h + Hnhf


H2s = Vector{Matrix}();
push!(H2s,H_i)
for ll = 2:N-2
	push!(H2s,H_m)
end
push!(H2s,H_f)


   
psi = smps.s3mps(N,chi); # create MPS
smps.ps!(psi, is0);  # create initial product state
	


dis1sm = Vector{ITensor}()   
for mm = 1:psi.N
    tmp = smps.op1(lm,psi.is[mm])
    push!(dis1sm,tmp)
end

dis2sm = Vector{ITensor}()   
for mm = 1:psi.N
    tmp = smps.op1(lp,psi.is[mm])
    push!(dis2sm,tmp)
end

dis1sz = Vector{ITensor}()   
for mm = 1:psi.N
    tmp = smps.op1(lz,psi.is[mm])
    push!(dis1sz,tmp)
end

dis2sp = Vector{ITensor}()   
for mm = 1:psi.N
    tmp = smps.op1(lp,psi.is[mm])
    push!(dis2sp,tmp)
end
exp1sm = Vector{ITensor}()
for mm = 1:psi.N
    tmp = smps.op1(lm'*lm,psi.is[mm])
    push!(exp1sm,tmp)
end

exp2sm = Vector{ITensor}()
for mm = 1:psi.N
    tmp = smps.op1(lp'*lp,psi.is[mm])
    push!(exp2sm,tmp)
end





	
cszs = Vector{ITensor}()
for jj = 1:N
    sz1 = smps.op1(sm'*sm, psi.is[jj]);
    push!(cszs,sz1)
end
    
id1s = Vector{ITensor}()   
for mm = 1:psi.N
    tmp = smps.op1(id2,psi.is[mm])
    push!(id1s,tmp)
end


function evolvejump1!(psi::smps.s3mps,op_1::Vector{ITensor}, op_2::Vector{ITensor},int::Int)
    expv = smps.expc1(psi,op_2)

    expv = expv./sum(expv)
    csp1 = cumsum(expv)
    dice2 = vecrand2[int]
    idxpick = findall(x -> x>dice2, csp1)
    choose = idxpick[1]
    choose = Int(choose)
    smps.apply1!(psi,op_1[choose]; nrm = true)
end

function evolvejump2!(psi::smps.s3mps,op_1::Vector{ITensor}, op_2::Vector{ITensor},op_3::Vector{ITensor},op_4::Vector{ITensor},int::Int)
    expv = smps.expc1(psi,op_3)
    expv2 = smps.expc1(psi,op_4)
    expv = vcat(expv,expv2)

    expv = expv./sum(expv)
    csp1 = cumsum(expv)
    dice2 = vecrand2[int]
    idxpick = findall(x -> x>dice2, csp1)
    choose = idxpick[1]
    choose = Int(choose)
    if choose > psi.N 
        choose = choose - psi.N
        smps.apply1!(psi,op_2[choose]; nrm = true)
    else
        smps.apply1!(psi,op_1[choose]; nrm = true)
    end
end

	
ctr = Int(round(N/2)+1)

csxs = Vector{ITensor}()
csys = Vector{ITensor}()
cszs = Vector{ITensor}()
cspms = Vector{ITensor}()
for ii = 1:N
	tmpx = smps.op1(sx, psi.is[ii]);
	tmpy = smps.op1(sy, psi.is[ii]);
	tmpz = smps.op1(sz, psi.is[ii]);
	tmppm = smps.op1(sm'*sm, psi.is[ii]);

	push!(csxs,tmpx)
	push!(csys,tmpy)
	push!(cszs,tmpz)
	push!(cspms,tmppm)
end
sweep = smps.trotter(psi, H2s, 2, -1im*dt)



let njump = 1
for tt = 1:steps
	tic = time();
	
	psnrm = real(smps.normps(psi)); 
    entr = smps.vne(psi, ctr)
	if isnan(entr)
		entr = 0.0
	else
	end
	psi1 = deepcopy(psi)
	
	#psnrm1 = real(norm(psi1));
	norm1 = smps.normps(psi1)
	smps.apply1!(psi1,id1s[1];nrm = true)
	
	
    esx = real(smps.expc1(psi1, csxs));
    esy = real(smps.expc1(psi1, csys));
    esz = real(smps.expc1(psi1, cszs));
	#println(esx[ctr])
    of = open(ofn, "a");
    println(of, (tt-1)*dt, ",", entr, ",", join(esz,","));
    close(of);

    
    if psnrm < vecrand[njump]
        evolvejump2!(psi,dis1sm,dis2sm,exp1sm,exp2sm,njump)
        njump +=1
        err = 0;
        for ss = 1:length(sweep[2])
            spec = smps.apply2!(psi, sweep[1][ss]; lr = sweep[2][ss]);
            err = max(spec.truncerr, err);
        end
    else
    	err = 0;
    	for ss = 1:length(sweep[2])
        	spec = smps.apply2!(psi, sweep[1][ss]; lr = sweep[2][ss]);
        	err = max(spec.truncerr, err)
    	end
	end

	
    toc = time()-tic;
    #println("step $tt/$steps -- elapsed $toc -- err=$err" )

end
end
end
#end
end
