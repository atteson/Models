using Ipopt

struct DistributionModel{D <: Distribution, U, T} <: AbstractModel{U,T}
    d::D
end

initialize( ::DistributionModel, args... ) = nothing

inequalityconstraintmatrix( ::Type{M} ) where {U,T,M <: DistributionModel{Normal,U,T}} = [0.0 1.0]
inequalityconstraintvector( ::Type{M} ) where {U,T,M <: DistributionModel{Normal,U,T}} = [0.0]

function rand( ::Type{M};
               t = domaintype(M)[],
               u = rangetype(M)[],
               seed = -1,
               kwargs... ) where {D,U,T,M <: DistributionModel{D,U,T}}
    if seed != -1
        Random.seed!( seed )
    end
    
    # A x <= b
    A = inequalityconstraintmatrix( M )
    b = inequalityconstraintvector( M )

    # thought about more general approaches but couldn't quite get there; issue is redudant equations
    indices = [findall(A[i,:] .!= 0) for i in 1:size(A,1)]
    @assert( all(length.(indices) .== 1) )
    indices = getindex.( indices, 1 )
    undices = unique(indices)
    @assert( length(undices) == length(indices) )
    parameters = Base.randn( size(A,2) )
    parameters[undices] = b .+ -log.(Base.rand(length(undices)))

    # this won't work for, for example, multivariate
    DistributionModel{D,U,T}( D( parameters... ) )
end

function rand!( model::DistributionModel{D,T,U}, t::AbstractVector{T}, u::AbstractVector{U}; seed::Int = -1 ) where {D,T,U}
    if seed != -1
        Random.seed!(seed)
    end
    Distributions.rand!( model.d, u )
end

loglikelihood( model::DistributionModel{D,T,U}, t::AbstractVector{T}, u::AbstractVector{U} ) where {D,T,U} =
    sum(log.(pdf.(model.d, u)))

function dloglikelihood( model::DistributionModel{Normal,T,U},
                         t::AbstractVector{T},
                         u::AbstractVector{U},
                         dl::AbstractVector{Float64} ) where {D,T,U}
    (mu,sigma) = params(model.d)
    dl[1] = sum(u .- mu)/sigma^2
    dl[2] = sum((u .- mu).^2)/sigma^3 - 1/sigma
end
                                                                                      
function fit( model::AbstractModel{T,U}, t::AbstractVector{T}, u::AbstractVector{U} ) where {T,U}
end

