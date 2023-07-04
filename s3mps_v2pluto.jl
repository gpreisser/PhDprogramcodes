### A Pluto.jl notebook ###
# v0.19.8

using Markdown
using InteractiveUtils

# ╔═╡ a6f2a2f3-c60c-4f32-9f5c-1ff9c5ecfcc2
using LinearAlgebra

# ╔═╡ c94a1f5a-0f15-4104-9b64-836d16c1cea9
using ITensors

# ╔═╡ 40cb5b73-02c5-41db-80fb-3972d224f2da
####################################################################################################
#
# Finite sized MPS structure
#
####################################################################################################


mutable struct s3mps

    N::Int      # number of sites
    chi::Int    # max bond dimension allowed

    ocen::Int               # site which is the orthonormality center
    gams::Vector{ITensor}   # gam 3-tensors

    is::Vector{Index}       # ITensor physical indices.
    chis::Vector{Index}    # ITensor bond indices (1 is leftmost)

    # constructor
    s3mps(N::Int, chi::Int) = new(N, chi, 1, Vector{ITensor}(), Vector{Index}(), Vector{Index}())

end

# ╔═╡ 99756793-eb84-4df8-a7ff-241b27306a6c
# +++ our ocen concept explained +++:
#
# In the MPS, all matrices to the left of ocen are left-orthogonal, i.e.
#
# psi.gams[nn] * prime(dag(psi.gams[nn], psi.chis[nn+1]) ~ id
#  ... for any nn < ocen
#
# At the same time all matrices to the right of ocen are right orthogonal, i.e.
#
# psi.gams[nn] * prime(dag(psi.gams[nn], psi.chis[nn]) ~ id
#  ... for any nn > ocen.
#
# This means that the gam tensor on site ocen can be manipulated, since it is expanded into an
# orthogonal basis on both sides.
#
# In the graphical representation from check_orth, this means:
#  ... (ocen-1)-x(ocen)x-(ocen+1) ... (of course the x could also be -)
# a "-" means that the bond is orthogonal on the side of the corresponding tensor. If it's on the left
# side the tensor is right orthogonal, if it's on the right side it's left orthogonal.
#
# If one multiplies the S from the svd into the right gam (lr = true)
# ... then only the left gam will be guaranteed to be left-orthogonal
#
# If one multiplies the S from the svd into the left gam (lr = false)
# ... then only the right gam will be guaranteed to be right-orthogonal







"""
Creates a product state with initial vector elements in "ampl".
ampl can be anything (state vector amplitudes, vectorized density matrices, Bloch vector elements, ...)
If the MPS is supposed to be an iMPS, set imps=true
"""
function ps!(psi::s3mps, ampl)

    if isa(ampl, Matrix)
        ampl = [cc[:] for cc in eachcol(ampl)]; # convert matrix to vector of vectors in case
    end

    # create physical indices
    psi.is = Vector([Index(length(ampl[nn]), string("i", nn)) for nn in 1:psi.N])

    # create bond indices
    psi.chis = Vector([Index(1, string("C", nn)) for nn in 1:(psi.N+1)])

    # create gam itensors
    for nn = 1:psi.N
        i = psi.is[nn];
        L = psi.chis[nn];
        R = psi.chis[nn+1]
        tmpg = ITensor(typeof(ampl[1][1]), i, L, R);
        for ii in 1:length(ampl[nn])
            tmpg[i(ii), L(1), R(1)] = ampl[nn][ii];
        end
        push!(psi.gams, tmpg);
    end

    # just per definition:
    psi.ocen = 1;

end


function ps2!(psi::s3mps, ampl)
    newvec = Vector{}();
    jj = 1;
    if isa(ampl, Matrix)
        for ii = 1:2:psi.N
            if isodd(jj)
                tmp = kron(ampl[:,ii],ampl[:,ii+1]); # convert matrix to vector of vectors in case
                push!(newvec,tmp)
            else
                tmp = ampl[:,ii]
                push!(newvec,tmp)
            end
            jj +=1
        end
    end
    # create physical indices
    psi.is = Vector([Index(length(newvec[nn]), string("i", nn)) for nn in 1:length(newvec)])

    # create bond indices
    psi.chis = Vector([Index(1, string("C", nn)) for nn in 1:(length(newvec)+1)])
    
    # create gam itensors
    for nn = 1:length(newvec)
        i = psi.is[nn];
        L = psi.chis[nn];
        R = psi.chis[nn+1]
        tmpg = ITensor(typeof(newvec[1][1]), i, L, R);
        for ii in 1:length(newvec[nn])
            tmpg[i(ii), L(1), R(1)] = newvec[nn][ii];
        end
        push!(psi.gams, tmpg);
    end

    # just per definition:
    psi.ocen = 1;

end
# ╔═╡ 1d63f6f3-4779-4c30-836f-f78ded09f897
"""
Creates a product state with quantum numbers. This version is for 1-d qn blocks only
Here, qns is a vector of quantum numbers on each site,
and iidx a vector, with the initial state choice on each site.
"""
function psc!(psi::s3mps, qnl::QN, qns::Vector{Vector{QN}}, iidxs::Array)

    # create physical indices
    psi.is = Vector([Index(qns[nn] .=> 1, string("i",nn)) for nn in 1:psi.N]);

    # create bond indices: by default indices go from left to right
    # each bond on the right of a tensor has less qn than on-site
    qnc = qnl;
    tmpidx = addtags(Index(qnc => 1), string("C",1));
    push!(psi.chis, tmpidx);

    for nn = 1:psi.N
        qnc -= qns[nn][iidxs[nn]];
        tmpidx = addtags(Index(qnc => 1), string("C",nn+1));
        push!(psi.chis, tmpidx);
    end

    # create gam itensors
    for nn = 1:psi.N
        i = psi.is[nn];
        L = psi.chis[nn];
        R = psi.chis[nn+1];
        tmpg = ITensor(i, dag(L), R);
        tmpg[iidxs[nn], 1, 1] = 1.0;
        push!(psi.gams, tmpg);
    end

    # just per definition:
    psi.ocen = 1;

end

# ╔═╡ eb163fd2-f4ae-43e1-873c-6d3e5ee6041a
"""
Shifts the ocen to site site
"""
function shift_ocen!(psi::s3mps, site::Int; algo::String="recursive")

    if site > psi.ocen

        for nn = psi.ocen:(site-1)

            # shift to right: Make the tensor nn left orthogonal
            gsvd = svd(psi.gams[nn], psi.is[nn], psi.chis[nn];
                        lefttags = string("C", nn+1),
                        alg = algo)

            psi.gams[nn] = gsvd.U;
            psi.chis[nn+1] = commonind(gsvd.U, gsvd.S);
            psi.gams[nn+1] *= gsvd.S*gsvd.V;
            psi.ocen +=1;

        end

    end

    if site < psi.ocen

        for nn = psi.ocen:-1:(site+1)

            # shift to left: Make the tensor nn right orthogonal
            gsvd = svd(psi.gams[nn], psi.chis[nn];
                        righttags = string("C", nn),
                        alg = algo)

            psi.gams[nn-1] *= gsvd.U*gsvd.S;
            psi.chis[nn] = commonind(gsvd.S, gsvd.V);
            psi.gams[nn] = gsvd.V;
            psi.ocen -= 1;

        end

    end

end

# ╔═╡ 939f53a6-d803-46d4-bdd0-89756e016fed
"""
Returns the expectation values for all sites of an ITensor.
In practice it is better to use locdm if multiple observables are wanted.
"""
function expc1(psi::s3mps, op::Vector{ITensor})::Vector{Float64}

    oocen = psi.ocen;

    expv = Vector{Float64}()
    for ii = 1:psi.N
        shift_ocen!(psi, ii);
        gamu = psi.gams[ii]
        gaml = dag(gamu);
        gamu *= op[ii];
        expc = noprime(gamu)*gaml;
        push!(expv,real(store(expc)[1]))
    end

    shift_ocen!(psi, oocen);
    return expv

end


function expcc1(psi::s3mps, op::Vector{ITensor})::Vector{Float64}

    oocen = psi.ocen;
    expv = Vector{Float64}()
    for ii = 1:psi.N
        shift_ocen!(psi, ii);
        gamu = psi.gams[ii]
        gaml = dag(gamu);
        gamu *= op[ii];
        expc = noprime(gamu)*gaml;
        push!(expv,real((expc)[1]))
    end
    shift_ocen!(psi, oocen);
    return expv

end
# ╔═╡ 29bb9141-5572-40da-a50f-e87069b08eb0
"""
Returns the expectation values <psi| ops[1+nn] ops[1] |psi>
over the range of ops. They and the MPS need to be in corresponding order
"""
function expc2(psi::s3mps, ops::Vector{ITensor})::Vector{}

    oocen = psi.ocen;

    idx = noprime(ind(ops[1],1));
    ii = findfirst(x -> x == idx, psi.is)

    expv = Vector{}()

    shift_ocen!(psi,ii)

    edg = prime(psi.gams[ii]* ops[1], psi.chis[ii+1]);
    edg *= prime(dag(psi.gams[ii]), psi.is[ii]);

    for nn = 2:length(ops)
        cc = ii+nn-1;

        edg *= prime(psi.gams[cc], psi.chis[cc]);

        tmp = edg*ops[nn];
        tmp *= prime(dag(psi.gams[cc]), psi.is[cc]);
        push!(expv, real(tmp[1]));

        if nn < length(ops)
            edg = prime(edg, psi.chis[cc+1])*dag(psi.gams[cc]);
        end

    end

    shift_ocen!(psi, oocen);
    return expv

end

# ╔═╡ 9a45d13d-4aa6-4e82-bb18-003347e1303f
"""
Returns the local density matrix on site "site" for a state-vector mps.
This can be used to evaluate local expectation values.
"""
function locdm(psi::s3mps, idx::Index)::Matrix

    site = findfirst(x -> x == idx, psi.is)

    oocen = psi.ocen;
    shift_ocen!(psi, site);

    tmp = psi.gams[site];
    tmp *= prime(dag(tmp), idx)

    dm = Matrix(tmp, inds(tmp)[1], inds(tmp)[2]);

    shift_ocen!(psi, oocen);

    return dm

end

# ╔═╡ 6837de93-6cbe-4a76-baf9-7e84f2733d05
"""
Swaps the two sites (site, site+1) in an MPS. Otherwise same options as apply2! except for nrm.

usage:

swap!(psi, site; lr=true, nrm=false, cutoff=0.0, algo="recursive")

- lr tells if afterwards the orthogonality center should be shifted to the left (0) or the right (1)
    It is not imperative to put it left or right, both ways will work, the function shifts ocen
    itself. It can save little computational time to put it to the correct side in a sweep.
- cutoff can be provided and gives the cutoff after which lam values are neglected
- algo is the svd algorithm (see e.g. ITensor svd doc):
    "divide_and_conquer" (faster, sometimes uinstable = LAPACK gesdd)
    "qr_iteration" (a bit slower, robust = LAPACK gesvd)
    "recursive" (ITensor implementation, reliable, sometimes slow for high precision)
"""
function swap!(psi::s3mps, site::Int;
                    lr::Bool=true,
                    cutoff::Float64=0.0,
                    algo::String="recursive")::Spectrum

    if  (psi.ocen != site) && (psi.ocen != (site+1))
        if psi.ocen < site
            shift_ocen!(psi,site,algo=algo)
        else
            shift_ocen!(psi,site+1,algo=algo)
        end
    end

    thet = psi.gams[site]*psi.gams[site+1];

    if lr
        newli = string("C", site+1)
        newri = "tc";
    else
        newli = "tc";
        newri = string("C", site+1);
    end

    tsvd = svd(thet, psi.is[site+1], psi.chis[site];
                maxdim = psi.chi,
                cutoff = cutoff, use_relative_cutoff = true,
                lefttags = newli, righttags = newri,
                alg = algo);

    # update
    if lr
        psi.chis[site+1] = commonind(tsvd.U, tsvd.S);
        psi.gams[site] = tsvd.U;
        psi.gams[site+1] = tsvd.V*tsvd.S;
        psi.ocen = site+1;
    else
        psi.chis[site+1] = commonind(tsvd.V, tsvd.S);
        psi.gams[site] = tsvd.U*tsvd.S;
        psi.gams[site+1] = tsvd.V;
        psi.ocen = site;
    end

    # now we  have to swap the site indices, the "chi" indices, remain and "keep entanglement"
    tmpi = psi.is[site];
    psi.is[site] = psi.is[site+1];
    psi.is[site+1] = tmpi;

    return tsvd.spec

end

# ╔═╡ b3081240-ae54-4ae9-a173-ea8561bcea5d
"""
returns the full norm of the s3mps by full contraction (for testing)
"""

# ╔═╡ e015f792-59ca-4b57-8f2c-49a22d12ba1b
"""
Applies a two-site gate. Truncation back to a max bond dimension of psi.chi.
Gates have to be in the form where the first index is on the left.
Works only for nearest neighbor gates.

usage:

apply2!(psi, gate2, nrm; lr=true, nrm=false, cutoff=0.0, algo="recursive")

- lr tells if afterwards the orthogonality center should be shifted to the left (0) or the right (1)
    It is not imperative to put it left or right, both ways will work, the function shifts ocen
    itself. It can save little computational time to put it to the correct side in a sweep.
- nrm is a boolean flag and tells is the two-site state should be re-normalized (euclidean norm) or not.
- cutoff can be provided and gives the cutoff after which lam values are neglected
- algo is the svd algorithm (see e.g. ITensor svd doc):
    "divide_and_conquer" (faster, sometimes uinstable = LAPACK gesdd)
    "qr_iteration" (a bit slower, robust = LAPACK gesvd)
    "recursive" (ITensor implementation, reliable, sometimes slow for high precision)
"""
function apply2!(psi::s3mps, gate2::ITensor;
                    lr::Bool=true,
                    nrm::Bool=false,
                    cutoff::Float64=0.0,
                    algo::String="recursive")::Spectrum

    pis = commoninds(noprime(inds(gate2))); # IndexSets are ordered sets, therefore pis[1] is the left index

    site = findfirst(x -> x == pis[1], psi.is);

    if  (psi.ocen != site) && (psi.ocen != (site+1))
        if psi.ocen < site
            shift_ocen!(psi,site,algo=algo)
        else
            shift_ocen!(psi,site+1,algo=algo)
        end
    end

    thet = psi.gams[site]*psi.gams[site+1];
    thet *= gate2;
    thet = noprime(thet);

    if nrm
        thet = thet/normps(thet);
    end

    if lr
        newli = string("C", site+1)
        newri = "tc";
    else
        newli = "tc";
        newri = string("C", site+1);
    end

    tsvd = svd(thet, psi.is[site], psi.chis[site];
                maxdim = psi.chi,
                cutoff = cutoff, use_relative_cutoff = true,
                lefttags = newli, righttags = newri,
                alg = algo);

    # update
    if lr
        psi.chis[site+1] = commonind(tsvd.U, tsvd.S);
        psi.gams[site] = tsvd.U;
        psi.gams[site+1] = tsvd.V*tsvd.S;
        psi.ocen = site+1;
    else
        psi.chis[site+1] = commonind(tsvd.V, tsvd.S);
        psi.gams[site] = tsvd.U*tsvd.S;
        psi.gams[site+1] = tsvd.V;
        psi.ocen = site;
    end

    return tsvd.spec

end

function apply3!(psi::s3mps, gate2::ITensor;
    lr::Bool=true,
    nrm::Bool=false,
    cutoff::Float64=0.0,
    algo::String="recursive")::Spectrum

pis = commoninds(noprime(inds(gate2))); # IndexSets are ordered sets, therefore pis[1] is the left index

site = findfirst(x -> x == pis[1], psi.is);

if  (psi.ocen != site) && (psi.ocen != (site+1))
if psi.ocen < site
shift_ocen!(psi,site,algo=algo)
else
shift_ocen!(psi,site+1,algo=algo)
end
end

thet = psi.gams[site]*psi.gams[site+1]*psi.gams[site+2];
thet *= gate2;

thet = noprime(thet);

if nrm
thet = thet/normps(thet);
end

if lr
newli = string("C", site+1)
newri = "tc";

else
newli = "tc";
newri = string("C", site+1);
end

tsvd = svd(thet, psi.is[site], psi.chis[site];
maxdim = psi.chi,
cutoff = cutoff, use_relative_cutoff = true,
lefttags = newli, righttags = newri,
alg = algo);

# update
if lr
psi.chis[site+1] = commonind(tsvd.U, tsvd.S);
psi.gams[site] = tsvd.U;
psi.gams[site+1] = tsvd.V*tsvd.S;
psi.ocen = site+1;
else
psi.chis[site+1] = commonind(tsvd.V, tsvd.S);
psi.gams[site] = tsvd.U*tsvd.S;
psi.gams[site+1] = tsvd.V;
psi.ocen = site;
end

if lr
newli2 = string("C", site+2)
newri2 = "tc";
else
newli2 = "tc";
newri2 = string("C", site+2);
end

tsvd2 = svd(tsvd.S*tsvd.V, psi.is[site+1], psi.chis[site+1];
maxdim = psi.chi,
cutoff = cutoff, use_relative_cutoff = true,
lefttags = newli2, righttags = newri2,
alg = algo);

# update
if lr
psi.chis[site+2] = commonind(tsvd2.U, tsvd2.S);
psi.gams[site+1] = tsvd2.U;
psi.gams[site+2] = tsvd2.V*tsvd2.S;
psi.ocen = site+2;
else
psi.chis[site+2] = commonind(tsvd.V, tsvd.S);
psi.gams[site+1] = tsvd.U*tsvd.S;
psi.gams[site+2] = tsvd.V;
psi.ocen = site;
end
return tsvd.spec

end



# ╔═╡ 52053156-4eda-4838-ab96-035cfa3c475c
"""
Applies a single-site operator which has to be a proper ITensor
"""
function apply1!(psi::s3mps, gate1::ITensor;
                    nrm::Bool=false)

    idx = noprime(ind(gate1,1));

    site = findfirst(x -> x == idx, psi.is)

    shift_ocen!(psi,site)

    psi.gams[site] = noprime(psi.gams[site]*gate1);

    if nrm
        psi.gams[site] = psi.gams[site]/norm(psi.gams[site]);
    end


end

# ╔═╡ db8d28be-3763-486b-96c5-68f0df7250b3
"""
Builds a partial trace (works e.g. for density matrix representations).
The local tracekets are needed.
"""
function partrace(psi::s3mps, idx::Index, trkets::Dict{Index,ITensor})::Vector


    site = findfirst(x -> x == idx, psi.is)

    edgl = delta(psi.chis[1])
    for nn = 1:(site-1)
        edgl *= psi.gams[nn]*trkets[psi.is[nn]]
    end

    edgr = delta(psi.chis[end])
    for nn = psi.N:-1:(site+1)
        edgr *= psi.gams[nn]*trkets[psi.is[nn]]
    end

    edgl *= psi.gams[site];
    edgl *= edgr

    return Vector(edgl,idx)

end

function partrace11(psi::s3mps, idx::Index, trkets::Vector{ITensor})


    site = findfirst(x -> x == idx, psi.is)
    edgl = delta(psi.chis[1])
    for nn = 1:(site-1)
        edgl *= psi.gams[nn]*trkets[nn]
    end
    
    edgr = delta(psi.chis[end])
    
    for nn = psi.N:-1:(site+1)
        edgr *= psi.gams[nn]*trkets[nn]
    end
    
    edgl *= psi.gams[site];
    edgl *= edgr

    return Vector(edgl,idx)

end


# ╔═╡ d4c38d55-d661-4dc3-b0ec-f4bca2cf7389
function partrace(psi::s3mps, idx1::Index, idx2::Index, trkets::Vector{ITensor})::Matrix
    edgl = delta(psi.chis[1])

    site1 = findfirst(x -> x == idx1, psi.is)
    site2 = findfirst(x -> x == idx2, psi.is)


    for nn = 1:(site1-1)
        edgl *= psi.gams[nn]*trkets[nn];
    end
    edgl = edgl*psi.gams[site1]

    for nn = site1+1:(site2-1)
        edgl *= psi.gams[nn]*trkets[nn];
    end
    edgl *= psi.gams[site2]

    edgr = delta(psi.chis[N+1])
    for nn = psi.N:-1:site2+1
        edgr *= psi.gams[nn]*trkets[nn];
    end

    edgl *= edgr

    return Matrix(edgl,idx1,idx2)

end

# ╔═╡ 86a1574a-daad-40a2-a9e0-fc7e629b29d0
"""
Returns the von Neumann entropy for a bond = between sites bond-1 and bond. It normalizes the lams before
"""
function vne(psi::s3mps, bond::Int; algo::String="recursive")::Float64

    oocen = psi.ocen;
    shift_ocen!(psi, bond)

    gsvd = svd(psi.gams[bond], psi.chis[bond];
                alg = algo)

    svec = zeros(dim(gsvd.S,1));
    for cc = 1:length(svec);
        svec[cc] = gsvd.S[cc,cc];
    end
    svec /= norm(svec);
    svec = svec.^2;
    svec = -svec.*log2.(svec);
    entr = sum(svec, dims=1)[1];

    shift_ocen!(psi, oocen)

    return entr;

end

# ╔═╡ 8609987b-fcc2-477b-b56c-30d4e2a4c167
"""
Returns the Rényi entropy for a bond = between sites bond-1 and bond, and a certain alpha. It normalizes the lams before
"""

# ╔═╡ 18f56aa1-3c4e-455e-bc10-f3860e504858
"""
Returns the Schmidt values for a bond = between sites bond-1 and bond. Returned as ITensor
"""
function schmidt(psi::s3mps, bond::Int; algo::String="recursive")::ITensor

    oocen = psi.ocen;
    shift_ocen!(psi, bond)
    gsvd = svd(psi.gams[bond], psi.chis[bond];
                alg = algo)
    shift_ocen!(psi, oocen)

    return gsvd.S;

end

# ╔═╡ f9888e9f-369a-45df-9b6f-69b2c64a4922
"""
Returns an array of length 2N+1,2 ([-N:1:N;],2) with the entropies for each block corresponding to a quantum number on a bond on the first column,
and the probability distribution in the second
"""
function entrqn(psi::s3mps, bond:: Int; algo::String="recursive")::Array{Float64,2}
    N = psi.N
    schs = schmidt(psi,bond);
    schs /= norm(schs)
    indsch = inds(schs)
    entrqns = zeros(Int(2*N + 1),2)
    for ll = 1:nnzblocks(schs)
        qn1 = qn(indsch[1],Block(ll))
        valqn = val(qn1,"M")
	ind = Int(N + valqn +1)
        tmp = diag(array(schs[Block(ll,ll)]))
        entrqns[ind,2] = norm(tmp).^2;
	tmp /= norm(tmp);
        tmp = tmp.^2;
        tmp = -tmp.*log2.(tmp);
        entr1 = sum(tmp)
      	entrqns[ind,1] = entr1
    end
    return entrqns
end

# ╔═╡ 0c47c614-91fe-4df7-90cd-91edc956b344
"""
Checks the orhtogonality for a full MPS
"""
function check_orth(psi::s3mps; tol::Float64=1e-12)
    for nn = 1:psi.N
        check_orth(psi, nn; tol=tol);
    end
end


function overlap(psi::s3mps,phi::s3mps)
    edg = delta(psi.chis[1],dag(phi.chis[1]))
    for nn = 1:(psi.N)
        edg *= psi.gams[nn];
        edg *= delta(psi.is[nn],dag(phi.is[nn]))
        edg *= dag(phi.gams[nn]);
    end
    edg *= delta(dag(phi.chis[end]),(psi.chis[end]))
    return edg[1]
end


# ╔═╡ 0d034551-01e5-45cb-a0eb-9f9455de0bea
####################################################################################################
#
# Infinite sized MPS structure
# TODO
####################################################################################################


mutable struct s3imps

    N::Int      # number of sites (for now fixed to 2)
    chi::Int    # max bond dimension allowed

    gams::Vector{ITensor}   # gam 3-tensors
    lams::Vector{ITensor}   # lam diagonal 2-tensors. Always (!) keep full canonical form (keep lam out of gam)
    is::Vector{Index}       # ITensor physical indices
    chisl::Vector{Index}     # ITensor bond indices (on left of gmas)
    chisr::Vector{Index}     # ITensor bond indices (on right of gams)
    lognorm::Float64         # Normalization: MPS = 10^lognorm * s3imps
    is_qnc::Bool               # Number conserving or not

    # constructor
    s3imps(N::Int, chi::Int) = new(N, chi, Vector{ITensor}(), Vector{ITensor}(), Vector{Index}(), Vector{Index}(), Vector{Index}(), 0.0, false)

end

# ╔═╡ f04dc66e-cf15-4f98-90b8-379547388c2a
# +++ the ocen concept used for infinite MPS +++:
#






"""
Creates a product state with initial vector elements in "ampl".
ampl can be anything (state vector amplitudes, vectorized density matrices, Bloch vector elements, ...)
This version is for an iMPS
"""
function ps!(psi::s3imps, ampl)

    if isa(ampl, Matrix)
        ampl = [cc[:] for cc in eachcol(ampl)]; # convert matrix to vector of vectors in case
    end

    # create physical indices
    psi.is = Vector([Index(length(ampl[nn]), string("i", nn)) for nn in 1:psi.N])

    # create bond indices
    psi.chisl = Vector([Index(1, string("CL", nn)) for nn in 1:(psi.N)])
    psi.chisr = Vector([Index(1, string("CR", nn)) for nn in 1:(psi.N)])

    # create gam itensors
    for nn = 1:psi.N
        i = psi.is[nn];
        L = psi.chisl[nn];
        R = psi.chisr[nn]
        tmpg = ITensor(typeof(ampl[1][1]), i, L, R);
        for ii in 1:length(ampl[nn])
            tmpg[i(ii), L(1), R(1)] = ampl[nn][ii];
        end
        push!(psi.gams, tmpg);
    end

    # create lam itensors
    push!(psi.lams, delta(psi.chisr[2],psi.chisl[1]));
    #for nn = 1:...
    #end
    push!(psi.lams, delta(psi.chisr[1],psi.chisl[2]));


end

# ╔═╡ a680038d-c2b2-41e3-b487-b06243914182
"""
Creates a product state with quantum numbers. This version is for 1-d qn blocks only
Here, qns is a vector assigning quantum numbers to the physical index components on each site,
and iidx a vector, with the initial state choice on each site.
"""
function psc!(psi::s3imps, qnl::QN, qns::Vector{Vector{QN}}, iidxs::Array)

    @assert psi.N == 2

    # create physical indices
    psi.is = Vector([Index(qns[nn] .=> 1, string("i",nn)) for nn in 1:psi.N]);

    # create bond indices: by default indices go from left to right
    # each bond on the right of a tensor has less qn than on-site
    qn1 = qnl;
    qn2 = qn1 - qns[1][iidxs[1]];
    # @assert qns[1][iidxs[1]] + qns[2][iidxs[2]] == 0

    cl1 = Index(qn1 => 1, tags="CL1");
    cl2 = Index(qn2 => 1, tags="CL2");
    cr1 = Index(qn2 => 1, tags="CR1");
    cr2 = Index(qn1 => 1, tags="CR2");

    psi.chisl = Vector([cl1, cl2])
    psi.chisr = Vector([cr1, cr2])

    # create gam itensors
    for nn = 1:psi.N
        i = psi.is[nn];
        L = psi.chisl[nn];
        R = psi.chisr[nn];
        tmpg = ITensor(i, dag(L), R);
        tmpg[i(iidxs[nn]), L(1), R(1)] = 1.0;
        push!(psi.gams, tmpg);
    end

    # create lam itensors
    push!(psi.lams, delta(dag(psi.chisr[2]),psi.chisl[1]));
    #for nn = 1:...
    #end
    push!(psi.lams, delta(dag(psi.chisr[1]),psi.chisl[2]));

    psi.is_qnc = true
end

# ╔═╡ 66f8aea3-7135-4eca-aeb5-6bf1f7c41645
function compute_VR(Gam::ITensor, lam::ITensor, cind::Index, rng; tol::Float64=1e-16)

    # cind = inds(lam)[2] # Index where psi is split
    VR = ITensor(Matrix(I(dim(cind))), dag(prime(cind)), prime(cind, 2))
    VR /= norm(VR)

    diff = 1
    i = 0
    nrm = 0
    while diff > tol || i < 5
        RVR = Gam * prime(lam, cind)
        RVR *= VR
        RVR *= prime(dag(lam), 2, cind)
        RVR *= prime(dag(Gam), cind)
        RVR = prime(RVR)
        # RVR = replaceinds(RVR, (chilr, prime(chilr)) => (chirr, prime(chirr)))
        nrm = norm(RVR)
        RVR = RVR / nrm

        diff = norm(RVR - VR)
        VR = RVR
        i += 1
        if i%500 == 0
            @warn "step $i, convergence still not reached"
            # println("Norm = $nrm, difference = $diff")
        end
    end

    # println("R VR = $nrm VR with error $diff after $i steps")

    return nrm, VR / sign(store(VR)[1])
end

# ╔═╡ 3f2375e7-a3fc-43a4-9248-7b35e05afc91
function compute_VL(Gam::ITensor, lam::ITensor, cind::Index, rng; tol::Float64=1e-16)

    # cind = inds(lam)[1] # Index where psi is split
    VL = ITensor(Matrix(I(dim(cind))), dag(prime(cind)), prime(cind, 2))
    VL /= norm(VL)

    diff = 1
    i = 0
    left_eigval = 1.
    while diff > tol || i < 5
        LVL = prime(lam, 2, cind) * Gam
        LVL = VL * LVL
        LVL = prime(dag(lam), cind) * LVL
        LVL = dag(Gam) * prime(LVL, cind)
        LVL = prime(LVL)
        # RVR = replaceinds(RVR, (chilr, prime(chilr)) => (chirr, prime(chirr)))
        left_eigval = norm(LVL)
        LVL = LVL / norm(LVL)

        diff = norm(LVL - VL)
        VL = LVL
        i += 1
        if i%500 == 0
            @warn "step $i, convergence still not reached"
            println("Difference = $diff")
        end
    end

    return left_eigval, VL / sign(store(VL)[1])
end

# ╔═╡ 13411481-1313-4f43-89df-eb297bd261e5
function apply_VR(TR::ITensor, VR::ITensor)::ITensor
    RVR = TR * VR
    RVR *= VR

    return RVR
end

# ╔═╡ e695010e-3bb5-440b-9527-d8ef8ae8b082
function tensqrtinv(A::ITensor, cutoff::Float64)
    Am = matrix(A)
    diagAm_fix = diag(Am) + cutoff * (abs.(diag(Am)) .< cutoff/2)
    sqrtAm = Matrix(Diagonal(sqrt.(diagAm_fix)))
    invsqrtAm = Matrix(Diagonal(1 ./ sqrt.(diagAm_fix)))
    sqrtA = ITensor(sqrtAm, inds(A))
    invsqrtA = ITensor(invsqrtAm, inds(A))

    return sqrtA, invsqrtA
end

# ╔═╡ b945ae0b-6cbd-4810-a3f1-17dabd3d04bb
function schmidtvalues(psi::s3imps, site)
    return store(psi.lams[site])
end

# ╔═╡ e60cdccf-1dcd-4989-bde5-7b7d128fb3e1
"""
Compute the partial trace for an iMPS of density matrices
"""
function partrace(psi::s3imps, idx::Index, trkets::Vector{ITensor})::Vector

    site = findfirst([idx == i for i in psi.is])

    # Single block
    singblock = prime(psi.lams[1], psi.chisr[end]) * (psi.gams[1]*trkets[1])
    singblock *= psi.lams[2]
    singblock *= (psi.gams[2] * trkets[2])
    singblockmat = Matrix(singblock, prime(psi.chisr[end]), psi.chisr[end])

    # Infinite blocks ... use just largest eigenvector for efficiency ?
    vals, vecs = eigen(singblockmat)
    # if !(all(abs(vals[end]) .>= abs.(vals)))
    #     println("Error: Last eigenvalue not largest!")
    #     println(vals[abs(vals[end]) .<= abs.(vals)])
    #     error("This should never happen!")
    # end
    maxind = findmax(abs.(vals))[2]
    if maxind != length(vals)
        @warn "Largest eigenvalue not positive"
        println(vals[abs(vals[end]) .<= abs.(vals)])
    end
    rvec = vecs[:,maxind]
    lvec = inv(vecs)[maxind,:]
    # Largest eigenvalue must be 1

    # Switch prime for correct contraction
    ledge = ITensor(lvec, psi.chisr[end])
    redge = ITensor(rvec, psi.chisr[end])

    blv = 10^psi.lognorm * ledge * psi.lams[1]

    if site == 1
        blv *= psi.gams[1]
        blv *= psi.lams[2]
        blv *= (psi.gams[2] * redge * trkets[2])
    else
        blv *= (psi.gams[1] * trkets[1])
        blv *= psi.lams[2]
        blv *= (psi.gams[2]*redge)
    end

    return vector(blv)
end

# ╔═╡ b0b038b4-8188-4db2-b3e3-a3669112cfdb
"""
Compute the partial trace for an iMPS of density matrices for 2 sites on idx and idx + dist
"""
function partrace2s(psi::s3imps, idx1::Index, dist::Int64, trkets::Vector{ITensor})::Matrix

    @assert dist > 0
    site = findfirst([idx1 == i for i in psi.is])
    site2 = 2 - (( site + dist ) % 2 )
    idx2 = psi.is[site2]

    # number of blocks between both sites
    nblocks = Int64(floor((dist - 1.99 + site - site2) / 2)) # (dist-2+site-site2)/2

    # Single block
    singblock = prime(psi.lams[1], psi.chisr[end]) * (psi.gams[1]*trkets[1])
    singblock *= psi.lams[2]
    singblock *= (psi.gams[2] * trkets[2])
    singblockmat = Matrix(singblock, prime(psi.chisr[end]), psi.chisr[end])

    # Infinite blocks
    vals, vecs = eigen(singblockmat)
    maxind = findmax(abs.(vals))[2]
    if maxind != length(vals)
        @warn "Largest eigenvalue not positive"
        println(vals[abs(vals[end]) .<= abs.(vals)])
    end
    rvec = vecs[:,maxind]
    lvec = inv(vecs)[maxind,:]
    # @assert all(abs(vals[end]) .>= abs.(vals))
    # rvec = vecs[:,end]
    # lvec = inv(vecs)[end,:]
    # Largest eigenvalue must be 1

    # Switch prime for correct contraction
    ledge = ITensor(lvec, psi.chisr[end])
    redge = ITensor(rvec, psi.chisr[end])

    # Start contracting from the left
    blv = 10^psi.lognorm * ledge * psi.lams[1]
    if nblocks == -1 # same block
        @assert site==1 && site2==2
        blv *= psi.gams[1]
        blv *= psi.lams[2]
        blv *= (prime(psi.gams[2], idx2) * redge)
    elseif nblocks == 0 # neighboring blocks
        if site == 1
            blv *= psi.gams[1]
            blv *= psi.lams[2]
            blv *= (psi.gams[2] * trkets[2])
        else # site == 2
            blv *= (psi.gams[1] * trkets[1])
            blv *= psi.lams[2]
            blv *= psi.gams[2]
        end
        blv *= 10^psi.lognorm * psi.lams[1]
        if site2 == 1
            blv *= prime(psi.gams[1], idx2)
            blv *= psi.lams[2]
            blv *= (psi.gams[2] * redge * trkets[2])
        else # site2 == 2
            blv *= (psi.gams[1] * trkets[1])
            blv *= psi.lams[2]
            blv *= (prime(psi.gams[2], idx2) * redge)
        end
    else # nblocks in between
        if site == 1
            blv *= psi.gams[1]
            blv *= psi.lams[2]
            blv *= (psi.gams[2] * trkets[2])
        else # site == 2
            blv *= (psi.gams[1] * trkets[1])
            blv *= psi.lams[2]
            blv *= psi.gams[2]
        end
        between_blocks = ITensor(vecs * Diagonal(vals.^nblocks) * inv(vecs),
                                 psi.chisr[2], prime(psi.chisr[2]))
        blv *= between_blocks / real(vals[end] ^ nblocks)
        blv *= 10^psi.lognorm * prime(psi.lams[1], psi.chisr[2])
        if site2 == 1
            blv *= prime(psi.gams[1], idx2)
            blv *= psi.lams[2]
            blv *= (psi.gams[2] * redge * trkets[2])
        else # site2 == 2
            blv *= (psi.gams[1] * trkets[1])
            blv *= psi.lams[2]
            blv *= (prime(psi.gams[2], idx2) * redge)
        end
    end

    return Matrix(blv, idx1, prime(idx2))
end

# ╔═╡ 1352cd0e-1ca1-4c0a-936a-a094e56302f5
function vne(psi::s3imps, bond::Int; algo::String="recursive", cutoff::Float64=1e-16)::Float64

    svec = max.(diag(matrix(psi.lams[bond])), cutoff)

    svec = normalize(svec) .^ 2;
    entr = sum(-svec .* log2.(svec));

    return entr
end

# ╔═╡ 8e9eed23-a5c8-47d2-b979-5a8c0df3c4d2
function ren(psi::s3mps, bond::Int, alpha::Float64; algo::String="recursive")::Float64
    if alpha == 1
	entr = vne(psi,bond)
    else
    	oocen = psi.ocen;
    	shift_ocen!(psi, bond)

    	gsvd = svd(psi.gams[bond], psi.chis[bond];
    	            alg = algo)

    	svec = zeros(dim(gsvd.S,1));
    	for cc = 1:length(svec);
    	    svec[cc] = gsvd.S[cc,cc];
    	end
    	svec /= norm(svec);
    	svec = svec.^2;
    	svec = svec.^alpha;
    	svec = sum(svec, dims=1)[1];
    	entr = log2.(svec)/(1-alpha);
    	shift_ocen!(psi, oocen)
    end
    return entr;

end

# ╔═╡ 116200b3-40a9-43c0-a427-761fd13ad654
function ren(psi::s3imps, bond::Int, alpha::Float64; algo::String="recursive", cutoff::Float64=1e-16)::Float64
    if alpha ==1
    	entr = vne(psi,bond)
    else
    	svec = max.(diag(matrix(psi.lams[bond])), cutoff)

    	svec = normalize(svec) .^ 2;
    	svec = svec .^ alpha;
    	entr = log2.(sum(svec))/(1-alpha);
    end
    return entr
end

# ╔═╡ fba1eb5c-5c72-4634-aa41-ce6d20872c92
"""
Returns an array of size (nnzblocks,3) with the quantum number in the first column, the corresponding in the second column,
and the probability in the third
"""
function entrqn(psi::s3imps, site::Int, name::String)::Array{Float64,2}
    @assert psi.is_qnc

    schs = psi.lams[site];
    schs /= norm(schs)
    indsch = inds(schs)
    data = zeros(nnzblocks(schs), 3)
    for ll = 1:nnzblocks(schs)
        qn1 = qn(indsch[1], ll)
        valqn = val(qn1, name)
        blockschmidt = diag(array(schs[Block(ll,ll)]))
        pqn = norm(blockschmidt)^2;
	    svec = normalize(blockschmidt) .^ 2;
        blockentr = sum( -svec .* log2.(svec))
      	data[ll, :] = [valqn pqn blockentr]
    end
    return data
end

# ╔═╡ 1baab005-eed5-4e79-9236-0010ff9796d9
####################################################################################################
#
# Helper functions for ITensors/Linear Algebra that don't rely on any structure:
#
####################################################################################################


"""
Return an inverted diagonal lam matrix
"""
function invlam(lam::ITensor)::ITensor


    ilam = diagITensor(dag(inds(lam)) );

    for nn = 1:size(lam,1)
        el = lam[nn,nn];
        #if abs(el) < 1e-20 # this would be a safety
        # ... but it is not be needed if the cutoff is always used in svd
        # ... in other words there should be never values < 1-16 in the lambdas ever.
        #    el = 0;
        #else
        el = inv(el);
        #end
        ilam[nn,nn] = el;
    end

    return ilam

end

# ╔═╡ 1a890e15-36c1-44a3-aa4b-cad808931943
# """
#     function reorthogonalize(psi::s3imps)

# Bring psi into its canonical form.
# """
# function reorthogonalize!(psi::s3imps; tol::Float64=1e-16)
#     @assert psi.N == 2
#     Gam = psi.gams[1] * psi.lams[2] * psi.gams[2]
#     lam = psi.lams[1]

#     VR  = prime(compute_VR(Gam, lam, tol=tol), -1)
#     VL  = prime(compute_VL(Gam, lam, tol=tol), -1)

#     D, W = eigen(VR, ishermitian=true)
#     sqrtD = ITensor(NDTensors.Diag(sqrt.(store(D))), inds(D))
#     X = W * sqrtD
#     xrind = Index(size(lam,1), tags="XR")
#     replaceinds!(X, inds(X), [inds(lam)[2], xrind])

#     D, W = eigen(VL, ishermitian=true)
#     sqrtD = ITensor(NDTensors.Diag(sqrt.(store(D))), inds(D))
#     Ydag = W * sqrtD
#     ytlind = Index(size(lam,1), tags="YtL")
#     Yt = ITensor(conj.(matrix(Ydag)), ytlind, inds(lam)[1])

#     tsvd = svd(Yt * lam * X, ytlind,
#                maxdim=psi.chi,
#                lefttags="CR2",
#                righttags="CL1")
#     Xinv = ITensor(inv(matrix(X)), inds(X))
#     Ytinv = ITensor(inv(matrix(Yt)), inds(Yt))

#     @warn("Check correct transpositions for computing Gamp")
#     Gamp = tsvd.V * Xinv * Gam * Ytinv * tsvd.U
#     lam1p = tsvd.S

#     psi.chisr[2] = inds(tsvd.U, "CR2")[1]
#     psi.chisl[1] = inds(tsvd.V, "CL1")[1]

#     thet = lam1p * prime(Gamp, psi.chisr[2])
#     thet *= prime(lam1p, psi.chisr[2])
#     t2svd = svd(thet, psi.chisr[2], psi.is[1], maxdim=psi.chi,
#                 lefttags="CR1",
#                 righttags="CL2")

#     ilam1 = invlam(lam1p)
#     lam2p = t2svd.S
#     gam1p = ilam1 * t2svd.U
#     gam2p = t2svd.V * ilam1

#     psi.chisr[1] = inds(t2svd.U, "CR1")[1]
#     psi.chisl[2] = inds(t2svd.V, "CL2")[1]

#     psi.lams = [lam1p, lam2p]
#     psi.gams = [gam1p, gam2p]

# end


"""
    Apply gate on site and site+1
"""
function dmapply2!(psi::s3imps, gate2::ITensor, site::Int64, rng;
                   nrm::Bool=false,
                   cutoff::Float64=1e-16,
                   cutoff2::Float64=1e-32,
                   algo::String="recursive",
                   tol=1e-14)::Spectrum

    @assert psi.N == 2
    site2 = (site%psi.N) + 1
    lam = psi.lams[site]
    Gam = psi.gams[site]*psi.lams[site2]*psi.gams[site2];

    Gam *= gate2;
    Gam = noprime(Gam);
    maxGam = maximum(abs.(store(Gam)));

    if nrm
        Gam = Gam/norm(Gam);
    else
        psi.lognorm += log10(maxGam);
        Gam = Gam / maxGam;
    end

    clind = psi.chisl[site]
    crind = psi.chisr[site2]

    # Reorthogonalize
    right_eval, VR  = compute_VR(Gam, lam, clind, rng, tol=tol)
    VR = prime(VR, -1)
    left_eval, VL  = compute_VL(Gam, lam, crind, rng, tol=tol)
    VL = prime(VL, -1)

    DR, WR = eigen(VR,
                   ishermitian=true,
                   cutoff=cutoff,
                   tags="XR")
    sqrtDR, sqrtDRinv = tensqrtinv(DR, cutoff)
    X = noprime(dag(WR) * sqrtDR)
    xrind = commonind(DR, WR)
    # xrind = Index(size(DR,1), tags="XR")
    # replaceinds!(X, inds(X, "Link,eigen"), [xrind])
    Xinv = noprime(dag(sqrtDRinv) * WR)
    # replaceinds!(Xinv, inds(Xinv, "Link,eigen"), [xrind])

    DL, WL = eigen(VL,
                 ishermitian=true,
                 cutoff=cutoff,
                 tags="YtL")
    ylind = commonind(DL, WL)
    sqrtDL, sqrtDLinv = tensqrtinv(DL, cutoff)
    Yt = noprime(dag(sqrtDL) * WL)
    Ytinv = noprime(dag(WL) * sqrtDLinv)

    tsvd = svd(Yt * lam * X, ylind,
               maxdim=psi.chi,
               cutoff=cutoff,
               lefttags="CR$site2",
               righttags="CL$site")

    Gamp = tsvd.V * Xinv
    Gamp *= Gam
    Gamp *= Ytinv
    Gamp *= tsvd.U

    lam1p = tsvd.S

    psi.chisr[site2] = inds(Gamp, "CR$site2")[1]
    psi.chisl[site] = inds(lam1p, "CL$site")[1]

    thet = lam1p * prime(Gamp, psi.chisr[site2])
    thet *= prime(lam1p, psi.chisr[site2])

    t2svd = svd(thet, psi.chisr[site2], psi.is[site],
                maxdim=psi.chi,
                cutoff=cutoff2,
                lefttags="CR$site",
                righttags="CL$site2")


    ilam1 = invlam(lam1p)
    lam2p = t2svd.S
    gam1p = ilam1 * t2svd.U
    gam2p = t2svd.V * ilam1

    psi.chisr[site] = inds(gam1p, "CR$site")[1]
    psi.chisl[site2] = inds(lam2p, "CL$site2")[1]


    maxlam1 = maximum(abs.(store(lam1p)))
    maxlam2 = maximum(abs.(store(lam2p)))
    maxgam1 = maximum(abs.(store(gam1p)))
    maxgam2 = maximum(abs.(store(gam2p)))

    psi.lams[[site,site2]] = [lam1p/maxlam1, lam2p/maxlam2]
    psi.gams[[site,site2]] = [gam1p/maxgam1, gam2p/maxgam2]

    psi.lognorm += log10(maxlam1 * maxlam2 * maxgam1 * maxgam2)

    return t2svd.spec

end

# ╔═╡ cedcfdbd-a867-4cf2-ae76-22941270189f
"""
Produce an ITensor from a vector "vec" for an index, e.g. useful for a traceket
"""
function vec1(vec::Vector, idx::Index)::ITensor

    ivec = ITensor(typeof(vec[1]), idx)

    for ii = 1:dim(idx)
        ivec[idx(ii)] = vec[ii];
    end

    return ivec
end

# ╔═╡ 213550ae-d35b-4c0b-87a5-0aed8e28cc36
"""
Produce an ITensor from a matrix "op" acting on an index
"""
function op1(op::Matrix, idx::Index)::ITensor


    i1 = prime(idx);
    j1 = dag(idx);

    top = ITensor(typeof(op[1]), (i1, j1))

    for ii = 1:dim(i1)
        for jj = 1:dim(j1)
            oval = op[ii,jj];
            if abs(oval) > 1e-16
                top[i1(ii), j1(jj)] = oval;
                # note that this will throw an error for missing number conservation in op.
            end

        end
    end

    return top

end

# ╔═╡ 87f01807-b3c3-43dc-b12c-1146510a1b70
"""
Produce an ITensor from a matrix "op". The function assumes a form of op that is a kronecker
product between two sites, where idx1 is the left index and idx2 is the right index.
"""
function op2(op::Matrix, idx1::Index, idx2::Index)::ITensor

    i1 = prime(idx1);
    j1 = dag(idx1);

    i2 = prime(idx2);
    j2 = dag(idx2);

    d1 = dim(j1);
    d2 = dim(j2);

    top = ITensor(typeof(op[1]), (i1, j1, i2, j2))

    op = reshape(op, (d2,d1,d2,d1));

    for ii1 = 1:d1
    for jj1 = 1:d1
    for ii2 = 1:d2
    for jj2 = 1:d2

            # for multiple-site matrices created from kron(), the site order is reversed.
            oval = op[ii2, ii1, jj2, jj1];
            if abs(oval) > 1e-16
                top[i1(ii1), j1(jj1), i2(ii2), j2(jj2)] = op[ii2, ii1, jj2, jj1];
                # note that this will throw an error for missing number conservation in op.
            end


    end
    end
    end
    end



    return top

end

"""
Produce an ITensor from a matrix "op". The function assumes a form of op that is a kronecker
product between two sites, where idx1 is the left index and idx2 is the right index.
"""
function op3(op::Matrix, idx1::Index, idx2::Index, idx3::Index)::ITensor

    i1 = prime(idx1);
    j1 = dag(idx1);

    i2 = prime(idx2);
    j2 = dag(idx2);

    i3 = prime(idx3);
    j3 = dag(idx3);

    d1 = dim(j1);
    d2 = dim(j2);
    d3 = dim(j3);

    top = ITensor(typeof(op[1]), (i1, j1, i2, j2, i3 , j3))

    op = reshape(op, (d3,d2,d1,d3,d2,d1));

    for ii1 = 1:d1
    for jj1 = 1:d1
    for ii2 = 1:d2
    for jj2 = 1:d2
    for ii3 = 1:d3
    for jj3 = 1:d3

            # for multiple-site matrices created from kron(), the site order is reversed.
            oval = op[ii3, ii2, ii1,jj3, jj2, jj1];
            if abs(oval) > 1e-16
                top[i1(ii1), j1(jj1), i2(ii2), j2(jj2), i3(ii3), j3(jj3)] = op[ii3, ii2, ii1, jj3, jj2, jj1];
                # note that this will throw an error for missing number conservation in op.
            end


    end
    end
    end
    end
    end
    end


    return top

end

function op32(op::Matrix, idx1::Index, idx2::Index)::ITensor

    i1 = prime(idx1);
    j1 = dag(idx1);
    print(i1)
    i2 = prime(idx2);
    j2 = dag(idx2);

    d1 = dim(j1)*2;
    d2 = dim(j2);

    top = ITensor(typeof(op[1]), (i1, j1, i2, j2))
    print(top)
    op = reshape(op, (d2,d1,d2,d1));
    print("got")
    for ii1 = 1:d1
    for jj1 = 1:d1
    for ii2 = 1:d2
    for jj2 = 1:d2

            # for multiple-site matrices created from kron(), the site order is reversed.
            oval = op[ii2, ii1, jj2, jj1];
            if abs(oval) > 1e-16
                top[i1(ii1), j1(jj1), i2(ii2), j2(jj2)] = op[ii2, ii1, jj2, jj1];
                print("finish here")
                # note that this will throw an error for missing number conservation in op.
            end


    end
    end
    end
    end



    return top

end



# ╔═╡ 76f06670-f82d-44fb-9b2e-0b06d599e840
"""
Takes an arbitrary matrix, removes values below tol,
and returns a real matrix if imaginary elements vanish
"""
function cleanmat(mat::Matrix; tol::Float64=1e-16)::Matrix

    rtmp = real(mat); rtmp[abs.(rtmp) .<  tol] .= 0;
    itmp = imag(mat); itmp[abs.(itmp) .< tol] .= 0;

    if iszero(itmp)
        mat = rtmp;
    else
        mat = rtmp + 1im*itmp;
    end

    return mat

end

# ╔═╡ 8be1f452-b6da-488e-ab91-444fd2fc2d75
"""
Returns trotter sweeps for a Hamiltonian decomposition into nearest neighbors, using the
Sorenborger-Stewart decompositions:
    https://arxiv.org/abs/quant-ph/9903055
Returns a tuple: sweep[1][:] are the gates, sweep[2][:] are the lr values for apply2!
"""
function trotter(psi::s3mps, H2s::Vector{Matrix}, order::Int, mimdt)::Tuple{Vector{ITensor}, Vector{Bool}}

    N = psi.N;

    if order == 2

        expmimdtH2 = Vector{ITensor}();  # build matrix exponentials
        for nn = 1:(N-1)
            tmp = cleanmat(exp((mimdt/2)*H2s[nn]));
            push!(expmimdtH2, op2(tmp , psi.is[nn], psi.is[nn+1]));
        end

        R = Vector{ITensor}();  # sweep to right
        for nn = 1:(N-2)
            push!(R, expmimdtH2[nn]);
        end

        E = prime(expmimdtH2[N-1]*prime(expmimdtH2[N-1]), -1, plev=2);

        L = Vector{ITensor}(); # sweep to left
        for nn = (N-2):-1:1
            push!(L, expmimdtH2[nn]);
        end

        sweep = [R; E; L];
        lrvec = [ones(Bool, length(R)); zeros(Bool, length(L)+1)]

    elseif order == 4

        expmimdtH12 = Vector{ITensor}();    # build matrix exponentials 1
        for nn = 1:(N-1)
            tmp = cleanmat(exp((mimdt/12)*H2s[nn]));
            push!(expmimdtH12, op2(tmp , psi.is[nn], psi.is[nn+1]));
        end

        exppimdtH6 = Vector{ITensor}();     # build matrix exponentials 2
        for nn = 1:(N-1)
            tmp = cleanmat(exp(-(mimdt/6)*H2s[nn]));
            push!(exppimdtH6, op2(tmp , psi.is[nn], psi.is[nn+1]));
        end

        R = Vector{ITensor}();  # sweep to right
        for nn = 1:(N-1)
            push!(R, expmimdtH12[nn]);
        end

        L = Vector{ITensor}(); # sweep to left
        for nn = (N-1):-1:1
            push!(L, expmimdtH12[nn]);
        end

        Rp = Vector{ITensor}();  # sweep to right
        for nn = 1:(N-1)
            push!(Rp, exppimdtH6[nn]);
        end

        Lp = Vector{ITensor}(); # sweep to left
        for nn = (N-1):-1:1
            push!(Lp, exppimdtH6[nn]);
        end


        sweep = [R; L; R; Lp; R; R; R; R; L; R; L; L; L; L; Rp; L; R; L];

        Rlr = ones(Bool, length(R));
        Llr = zeros(Bool, length(L));
        lrvec = [Rlr; Llr; Rlr; Llr; Rlr; Rlr; Rlr; Rlr; Llr; Rlr; Llr; Llr; Llr; Llr; Rlr; Llr; Rlr; Llr];



    else
        error("Order $order not implemented.")
    end


    return (sweep, lrvec)

end


"""
Returns trotter sweeps for a Hamiltonian decomposition into nearest neighbors, using the
Sorenborger-Stewart decompositions:
    https://arxiv.org/abs/quant-ph/9903055
Returns a tuple: sweep[1][:] are the gates, sweep[2][:] are the lr values for apply2!
"""
function trotter3n(psi::s3mps, H2s::Vector{Matrix}, order::Int, mimdt)::Tuple{Vector{ITensor}, Vector{Bool}}

    N = psi.N;

    if order == 2

        expmimdtH2 = Vector{ITensor}();  # build matrix exponentials
        for nn = 1:(N-2)
            tmp = cleanmat(exp((mimdt/2)*H2s[nn]));
            push!(expmimdtH2, op3(tmp , psi.is[nn], psi.is[nn+1], psi.is[nn+2]));
        end

        R = Vector{ITensor}();  # sweep to right
        for nn = 1:(N-3)
            push!(R, expmimdtH2[nn]);
        end

        E = prime(expmimdtH2[N-2]*prime(expmimdtH2[N-2]), -1, plev=2);

        L = Vector{ITensor}(); # sweep to left
        for nn = (N-3):-1:1
            push!(L, expmimdtH2[nn]);
        end

        sweep = [R; E; L];
        lrvec = [ones(Bool, length(R)); zeros(Bool, length(L)+1)]

    elseif order == 4

        expmimdtH12 = Vector{ITensor}();    # build matrix exponentials 1
        for nn = 1:(N-1)
            tmp = cleanmat(exp((mimdt/12)*H2s[nn]));
            push!(expmimdtH12, op2(tmp , psi.is[nn], psi.is[nn+1]));
        end

        exppimdtH6 = Vector{ITensor}();     # build matrix exponentials 2
        for nn = 1:(N-1)
            tmp = cleanmat(exp(-(mimdt/6)*H2s[nn]));
            push!(exppimdtH6, op2(tmp , psi.is[nn], psi.is[nn+1]));
        end

        R = Vector{ITensor}();  # sweep to right
        for nn = 1:(N-1)
            push!(R, expmimdtH12[nn]);
        end

        L = Vector{ITensor}(); # sweep to left
        for nn = (N-1):-1:1
            push!(L, expmimdtH12[nn]);
        end

        Rp = Vector{ITensor}();  # sweep to right
        for nn = 1:(N-1)
            push!(Rp, exppimdtH6[nn]);
        end

        Lp = Vector{ITensor}(); # sweep to left
        for nn = (N-1):-1:1
            push!(Lp, exppimdtH6[nn]);
        end


        sweep = [R; L; R; Lp; R; R; R; R; L; R; L; L; L; L; Rp; L; R; L];

        Rlr = ones(Bool, length(R));
        Llr = zeros(Bool, length(L));
        lrvec = [Rlr; Llr; Rlr; Llr; Rlr; Rlr; Rlr; Rlr; Llr; Rlr; Llr; Llr; Llr; Llr; Rlr; Llr; Rlr; Llr];



    else
        error("Order $order not implemented.")
    end


    return (sweep, lrvec)

end



function trotter3(psi::s3mps, H2s::Vector{Matrix}, order::Int, mimdt)::Tuple{Vector{ITensor}, Vector{Bool}}

    N = psi.N;

    if order == 2

        expmimdtH2 = Vector{ITensor}();  # build matrix exponentials
        for nn = 1:(N-2)
            tmp = cleanmat(exp((mimdt/2)*H2s[nn]));
            push!(expmimdtH2, op3(tmp , psi.is[nn], psi.is[nn+1],psi.is[nn+2]));
        end
       
            
        R = Vector{ITensor}();  # sweep to right
        for nn = 1:(N-3)
            push!(R, expmimdtH2[nn]);
        end

        E = prime(expmimdtH2[N-2]*prime(expmimdtH2[N-2]), -1, plev=2);

        L = Vector{ITensor}(); # sweep to left
        for nn = (N-3):-1:1
            push!(L, expmimdtH2[nn]);
        end
        if N == 3
            sweep = [E;];
            lrvec = [ones(Bool, 1);]
        else
            sweep = [R;E; L];
            lrvec = [ones(Bool, length(R)); zeros(Bool, length(L)+1)]
        end

    elseif order == 4

        expmimdtH12 = Vector{ITensor}();    # build matrix exponentials 1
        for nn = 1:(N-2)
            tmp = cleanmat(exp((mimdt/12)*H2s[nn]));
            push!(expmimdtH12, op2(tmp , psi.is[nn], psi.is[nn+1]));
        end

        exppimdtH6 = Vector{ITensor}();     # build matrix exponentials 2
        for nn = 1:(N-2)
            tmp = cleanmat(exp(-(mimdt/6)*H2s[nn]));
            push!(exppimdtH6, op2(tmp , psi.is[nn], psi.is[nn+1]));
        end

        R = Vector{ITensor}();  # sweep to right
        for nn = 1:(N-2)
            push!(R, expmimdtH12[nn]);
        end

        L = Vector{ITensor}(); # sweep to left
        for nn = (N-2):-1:1
            push!(L, expmimdtH12[nn]);
        end

        Rp = Vector{ITensor}();  # sweep to right
        for nn = 1:(N-2)
            push!(Rp, exppimdtH6[nn]);
        end

        Lp = Vector{ITensor}(); # sweep to left
        for nn = (N-2):-1:1
            push!(Lp, exppimdtH6[nn]);
        end


        sweep = [R; L; R; Lp; R; R; R; R; L; R; L; L; L; L; Rp; L; R; L];

        Rlr = ones(Bool, length(R));
        Llr = zeros(Bool, length(L));
        lrvec = [Rlr; Llr; Rlr; Llr; Rlr; Rlr; Rlr; Rlr; Llr; Rlr; Llr; Llr; Llr; Llr; Rlr; Llr; Rlr; Llr];



    else
        error("Order $order not implemented.")
    end


    return (sweep, lrvec)

end

function trotter3o(psi::s3mps, H2s::Vector{Matrix}, order::Int, pair::Bool,mimdt)::Tuple{Vector{ITensor}, Vector{Bool}}

    N = psi.N;
    N1 = length(H2s);
    if pair==true
        global btrott = 2
    else
        global btrott = 1
    end
    if order == 2

        expmimdtH2 = Vector{ITensor}();  # build matrix exponentials
        ll = 1;
        for nn = btrott:2:(N-2)
            tmp = cleanmat(exp((mimdt/2)*H2s[ll]));
            push!(expmimdtH2, op3(tmp , psi.is[nn], psi.is[nn+1],psi.is[nn+2]));
            ll +=1;
        end
       
            
        R = Vector{ITensor}();  # sweep to right
        for nn = 1:N1-1
            push!(R, expmimdtH2[nn]);
        end

        E = prime(expmimdtH2[N1]*prime(expmimdtH2[N1]), -1, plev=2);

        L = Vector{ITensor}(); # sweep to left
        for nn = (N1-1):-1:1
            push!(L, expmimdtH2[nn]);
        end
        if N == 3
            sweep = [E;];
            lrvec = [ones(Bool, 1);]
        else
            sweep = [R;E; L];
            lrvec = [ones(Bool, length(R)); zeros(Bool, length(L)+1)]
        end

    elseif order == 4

        expmimdtH12 = Vector{ITensor}();    # build matrix exponentials 1
        for nn = 1:(N-2)
            tmp = cleanmat(exp((mimdt/12)*H2s[nn]));
            push!(expmimdtH12, op2(tmp , psi.is[nn], psi.is[nn+1]));
        end

        exppimdtH6 = Vector{ITensor}();     # build matrix exponentials 2
        for nn = 1:(N-2)
            tmp = cleanmat(exp(-(mimdt/6)*H2s[nn]));
            push!(exppimdtH6, op2(tmp , psi.is[nn], psi.is[nn+1]));
        end

        R = Vector{ITensor}();  # sweep to right
        for nn = 1:(N-2)
            push!(R, expmimdtH12[nn]);
        end

        L = Vector{ITensor}(); # sweep to left
        for nn = (N-2):-1:1
            push!(L, expmimdtH12[nn]);
        end

        Rp = Vector{ITensor}();  # sweep to right
        for nn = 1:(N-2)
            push!(Rp, exppimdtH6[nn]);
        end

        Lp = Vector{ITensor}(); # sweep to left
        for nn = (N-2):-1:1
            push!(Lp, exppimdtH6[nn]);
        end


        sweep = [R; L; R; Lp; R; R; R; R; L; R; L; L; L; L; Rp; L; R; L];

        Rlr = ones(Bool, length(R));
        Llr = zeros(Bool, length(L));
        lrvec = [Rlr; Llr; Rlr; Llr; Rlr; Rlr; Rlr; Rlr; Llr; Rlr; Llr; Llr; Llr; Llr; Rlr; Llr; Rlr; Llr];



    else
        error("Order $order not implemented.")
    end


    return (sweep, lrvec)

end

function trotter3ob(psi::s3mps, H2s::Vector{Matrix}, order::Int, pair::Bool,mimdt)::Tuple{Vector{ITensor}, Vector{Bool}}

    N = psi.N;
    N1 = length(H2s);
    if pair==true
        global btrott = 2
    else
        global btrott = 1
    end
    if order == 2

        expmimdtH2 = Vector{ITensor}();  # build matrix exponentials
        ll = 1;
        for nn = btrott:2:(N-2)
            tmp = cleanmat(exp((mimdt)*H2s[ll]));
            push!(expmimdtH2, op3(tmp , psi.is[nn], psi.is[nn+1],psi.is[nn+2]));
            ll +=1;
        end
       
            
        R = Vector{ITensor}();  # sweep to right
        for nn = 1:N1
            push!(R, expmimdtH2[nn]);
        end

        E = prime(expmimdtH2[N1]*prime(expmimdtH2[N1]), -1, plev=2);

        #L = Vector{ITensor}(); # sweep to left
        #for nn = (N1-1):-1:1
            #push!(L, expmimdtH2[nn]);
        #end
        if N == 3
            sweep = [E;];
            lrvec = [ones(Bool, 1);]
        else
            sweep = [R;];
            lrvec = [ones(Bool, length(R));]
        end

    elseif order == 4

        expmimdtH12 = Vector{ITensor}();    # build matrix exponentials 1
        for nn = 1:(N-2)
            tmp = cleanmat(exp((mimdt/12)*H2s[nn]));
            push!(expmimdtH12, op2(tmp , psi.is[nn], psi.is[nn+1]));
        end

        exppimdtH6 = Vector{ITensor}();     # build matrix exponentials 2
        for nn = 1:(N-2)
            tmp = cleanmat(exp(-(mimdt/6)*H2s[nn]));
            push!(exppimdtH6, op2(tmp , psi.is[nn], psi.is[nn+1]));
        end

        R = Vector{ITensor}();  # sweep to right
        for nn = 1:(N-2)
            push!(R, expmimdtH12[nn]);
        end

        L = Vector{ITensor}(); # sweep to left
        for nn = (N-2):-1:1
            push!(L, expmimdtH12[nn]);
        end

        Rp = Vector{ITensor}();  # sweep to right
        for nn = 1:(N-2)
            push!(Rp, exppimdtH6[nn]);
        end

        Lp = Vector{ITensor}(); # sweep to left
        for nn = (N-2):-1:1
            push!(Lp, exppimdtH6[nn]);
        end


        sweep = [R; L; R; Lp; R; R; R; R; L; R; L; L; L; L; Rp; L; R; L];

        Rlr = ones(Bool, length(R));
        Llr = zeros(Bool, length(L));
        lrvec = [Rlr; Llr; Rlr; Llr; Rlr; Rlr; Rlr; Rlr; Llr; Rlr; Llr; Llr; Llr; Llr; Rlr; Llr; Rlr; Llr];



    else
        error("Order $order not implemented.")
    end


    return (sweep, lrvec)

end

function trotter3a(psi::s3mps, H2s::Vector{Matrix}, order::Int, mimdt)::Tuple{Vector{ITensor}, Vector{Bool}}

    N = psi.N;
    N1 = length(H2s);
    if order == 2

        expmimdtH2 = Vector{ITensor}();  # build matrix exponentials
        for nn = 1:2:(N-2)
            tmp = cleanmat(exp((mimdt/2)*H2s[nn]));
            push!(expmimdtH2, op3(tmp , psi.is[nn], psi.is[nn+1],psi.is[nn+2]));
        end
       
            
        R = Vector{ITensor}();  # sweep to right
        for nn = 1:N1-1
            push!(R, expmimdtH2[nn]);
        end

        E = prime(expmimdtH2[N1]*prime(expmimdtH2[N1]), -1, plev=2);

        L = Vector{ITensor}(); # sweep to left
        for nn = (N1-1):-1:1
            push!(L, expmimdtH2[nn]);
        end
        if N == 3
            sweep = [E;];
            lrvec = [ones(Bool, 1);]
        else
            sweep = [R;E; L];
            lrvec = [ones(Bool, length(R)); zeros(Bool, length(L)+1)]
        end

    elseif order == 4

        expmimdtH12 = Vector{ITensor}();    # build matrix exponentials 1
        for nn = 1:(N-2)
            tmp = cleanmat(exp((mimdt/12)*H2s[nn]));
            push!(expmimdtH12, op2(tmp , psi.is[nn], psi.is[nn+1]));
        end

        exppimdtH6 = Vector{ITensor}();     # build matrix exponentials 2
        for nn = 1:(N-2)
            tmp = cleanmat(exp(-(mimdt/6)*H2s[nn]));
            push!(exppimdtH6, op2(tmp , psi.is[nn], psi.is[nn+1]));
        end

        R = Vector{ITensor}();  # sweep to right
        for nn = 1:(N-2)
            push!(R, expmimdtH12[nn]);
        end

        L = Vector{ITensor}(); # sweep to left
        for nn = (N-2):-1:1
            push!(L, expmimdtH12[nn]);
        end

        Rp = Vector{ITensor}();  # sweep to right
        for nn = 1:(N-2)
            push!(Rp, exppimdtH6[nn]);
        end

        Lp = Vector{ITensor}(); # sweep to left
        for nn = (N-2):-1:1
            push!(Lp, exppimdtH6[nn]);
        end


        sweep = [R; L; R; Lp; R; R; R; R; L; R; L; L; L; L; Rp; L; R; L];

        Rlr = ones(Bool, length(R));
        Llr = zeros(Bool, length(L));
        lrvec = [Rlr; Llr; Rlr; Llr; Rlr; Rlr; Rlr; Rlr; Llr; Rlr; Llr; Llr; Llr; Llr; Rlr; Llr; Rlr; Llr];



    else
        error("Order $order not implemented.")
    end


    return (sweep, lrvec)

end


# ╔═╡ 0cf1bc4e-6c9b-4d3a-9f4d-1585049cdc87
"""
Checks the orhtogonality at a site with a tolerance of zero values.
Prints a graphical representation and returns the "supposed identities"
"""
function check_orth(psi::s3mps, site::Int; tol::Float64=1e-12)::Vector{Matrix}

    mats = Vector{Matrix}(undef, 2);

    R = psi.gams[site]*prime(dag(psi.gams[site]),psi.chis[site])
    L = psi.gams[site]*prime(dag(psi.gams[site]),psi.chis[site+1])

    Rmat = cleanmat(Matrix(R, inds(R)[1], inds(R)[2]), tol=tol);
    Lmat = cleanmat(Matrix(L, inds(L)[1], inds(L)[2]), tol=tol);

    rsym = "x";
    if isdiag(Rmat)
        rsym = "-";
    end
    lsym = "x";
    if isdiag(Lmat)
        lsym = "-";
    end

    print(string(rsym,site,lsym))

    mats[1] = Rmat;
    mats[2] = Lmat;

    return mats;

end

# ╔═╡ c927fcb2-e96f-4c2a-b424-22976b14e206
"""
check orthogonality of the iMPS on site site
"""
function check_orth(psi::s3imps, site::Int; tol::Float64=1e-12)::Vector{Matrix}

    mats = Vector{Matrix}(undef, 2);

    rsite = (site%psi.N)+1;

    R = psi.gams[site]*psi.lams[rsite];
    R *= prime(dag(R), psi.chisl[site])

    L = psi.lams[site]*psi.gams[site];
    L *= prime(dag(L),psi.chisr[site])

    Rmat = cleanmat(Matrix(R, inds(R)[1], inds(R)[2]), tol=tol);
    Lmat = cleanmat(Matrix(L, inds(L)[1], inds(L)[2]), tol=tol);

    rsym = "x";
    if isdiag(Rmat)
        rsym = "-";
    end
    lsym = "x";
    if isdiag(Lmat)
        lsym = "-";
    end

    print(string(rsym,site,lsym))

    mats[1] = Rmat;
    mats[2] = Lmat;

    return mats;

end

# ╔═╡ 24dc9b24-e5a8-46ef-93a6-0f0e0842613f
import LinearAlgebra.tr

# ╔═╡ 81c74b98-6106-4881-b4bc-023118b41857
function normps(psi::s3mps)

    edg = delta(psi.chis[1],prime(dag(psi.chis[1])))

    for nn = 1:(psi.N)

        edg *= psi.gams[nn];
        edg *= prime(dag(psi.gams[nn]), psi.chis[nn], psi.chis[nn+1]);

    end

    edg *= delta(dag(psi.chis[end]),prime(psi.chis[end]))

    return edg[1]

end

# ╔═╡ 7e82d87d-03e7-4a65-917e-777f356832db
import LinearAlgebra.norm

# ╔═╡ ae92af35-a58c-42cf-b8a3-2bd740f9000e
"""
returns the *full* trace of the s3mps in case of a dm representation (may be a good error estimate)
"""
function tr(psi::s3mps, trkets::Dict{Index,ITensor})::Float64

    edg = delta(psi.chis[1])

    for nn = 1:(psi.N)

        edg *= psi.gams[nn]*trkets[psi.is[nn]];

    end

    edg *= delta(psi.chis[end])

    return store(edg)[1]

end

