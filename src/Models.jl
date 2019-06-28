module Models

using Dates
using Distributions
using Random
using Distributed
using Dependencies

abstract type AbstractModel{T,U}
end

function update( M::AbstractModel{T}, t::AbstractVector{T}, u::AbstractVector{U}; kwargs... ) where {T,U}
    for i = 1:length(t)
        update( M, t[i], u[i]; kwargs... )
    end
end

function Base.rand( M::AbstractModel{T,U}, t::AbstractVector{T}; kwargs... ) where {T,U}
    observations = zeros(U,length(t))
    Distributions.rand!( M, t, observations; kwargs... )
    return observations
end

domaintype( ::Type{V} ) where {T,U,V <: AbstractModel{T,U}} = T

rangetype( ::Type{V} ) where {T,U,V <: AbstractModel{T,U}} = U

function Base.rand( M::Type{V}; t = domaintype(V)[], u = rangetype(V)[], kwargs... ) where {T,U,V <: AbstractModel{T,U}}
    model = rand( M; kwargs... )
    initialize( model, t, u )
    return model
end

sandwich( M::AbstractModel ) = sandwich( rootmodel( M ) )

getcompressedparameters( M::AbstractModel ) = getcompressedparameters( rootmodel( M ) )

setcompressedparameters!( M::AbstractModel, p::AbstractVector{Float64} ) = setcompressedparameters!( rootmodel( M ), p )

struct FittableModel{T, U, V <: AbstractModel{T,U}, F <: Function} <: AbstractModel{T,U}
    model::V
    f::F
end

fit( model::FittableModel{T,U,V,F}; kwargs... ) where {T,U,V,F} = FittableModel( model.f( model.model; kwargs... ), model.f )

update( model::FittableModel{T,U,V,F}, t::T, u::U ) where {T,U,V,F} = update( model.model, t, u )

Dependencies.getinstance( ::Type{F} ) where {F <: Function} = F.instance

rand( ::Type{FittableModel{T,U,V,F}}; fitfunction::F = Dependencies.getinstance( F ), kwargs... ) where {T, U, V, F} =
    FittableModel( rand( V; kwargs... ), fitfunction )

Distributions.rand!( model::FittableModel{T,U,V,F}, t::AbstractVector{T}, u::AbstractVector{U} ) where {T,U,V,F} =
    rand!( model.model, t, u )

Dependencies.compress( model::FittableModel{T,U,V,F} ) where {T,U,V,F} = Dependencies.compress( model.model )

initialize( model::FittableModel{T,U,V,F}, args... ) where {T,U,V,F} = initialize( model.model, args... )

initializationcount( ::Type{FittableModel{T,U,V,F}} ) where {T,U,V,F} = initializationcount( V )

state( model::FittableModel{T,U,V,F} ) where {T,U,V,F} = state( model.model )

rootmodel( model::FittableModel{T,U,V,F} ) where {T,U,V,F} = rootmodel( model.model )

Base.rand( model::FittableModel{T,U,V,F} ) where {T,U,V,F} = Base.rand( model.model )

mutable struct LogReturnModel{T,V <: AbstractModel{T,Float64}} <: AbstractModel{T,Float64}
    model::V
    lastprice::Vector{Float64}
end

function update( model::LogReturnModel{T,V}, t::T, u::Float64 ) where {T,V}
    update( model.model, t, log(u/model.lastprice[1]) )
    model.lastprice[1] = u
end

function Distributions.rand!( model::LogReturnModel{T,V}, t::AbstractVector{T}, u::AbstractVector{Float64} ) where {T,V}
    rand!( model.model, t, u )
    lastprice = model.lastprice[1]
    for i = 1:length(t)
        u[i] = lastprice *= exp( u[i] )
    end
end

rand( ::Type{LogReturnModel{T,V}}; kwargs... ) where {T,V} = LogReturnModel( rand( V; kwargs... ), Float64[] )

fit( model::LogReturnModel; kwargs... ) = LogReturnModel( fit( model.model; kwargs... ), model.lastprice )

Dependencies.compress( model::LogReturnModel{T,V} ) where {T,V} = Dependencies.compress( model.model )

initializationcount( ::Type{LogReturnModel{T,V}} ) where {T,V} = max(initializationcount( V ), 1)

function initialize( model::LogReturnModel{T,V}, t::AbstractVector{T} = T[], u::AbstractVector{Float64} = Float64[] ) where {T,V}
    model.lastprice = u[end:end]
    initialize( model.model, t, u )
end

state( model::LogReturnModel{T,V} ) where {T,V} = state( model.model )

rootmodel( model::LogReturnModel{T,V} ) where {T,V} = rootmodel( model.model )

Base.rand( model::LogReturnModel{T,V} ) where {T,V} = Base.rand( model.model )

mutable struct MultiStartModel{T, U, V <: AbstractModel{T,U}, F <: Function} <: AbstractModel{T,U}
    models::Vector{V}
    criterion::F
    optimumindex::Int
end

function rand(
    ::Type{MultiStartModel{T,U,V,F}};
    seeds::AbstractVector{Int} = 1:1,
    kwargs...
) where {T, U, V <: AbstractModel{T,U}, F}
    models = V[]
    for seed in seeds
        Random.seed!( seed )
        push!( models, rand( V; kwargs... ) )
    end
    return MultiStartModel( models, Dependencies.getinstance( F ), 0 )
end

function update( model::MultiStartModel{T,U,V,F}, t::T, u::U ) where {T,U,V,F}
    for submodel in model.models
        update( submodel, t, u )
    end
end

function fit(
    model::MultiStartModel{T,U,V,F};
    modules::Vector{Symbol} = Symbol[],
    kwargs...
) where {T,U,V,F}
    if nprocs() == 1
        models = [fit( model.models[1]; kwargs... )]
        for i = 2:length(model.models)
            push!( models, fit( model.models[i]; kwargs... ) )
        end
    else
        # I don't know of a more convenient way to load all the modules we want
        futures = Future[]
        for pid in workers()
            for moduletoeval in modules
                push!( futures, remotecall( Core.eval, pid, Main, Expr(:using,Expr(:.,moduletoeval)) ) )
            end
        end
        for future in futures
            wait(future)
        end

        models = pmap( submodel -> fit( submodel; kwargs... ), model.models )
    end
    criteria = model.criterion.( model.models )
    optimumindex = findmax( criteria )[2]
    return MultiStartModel( models, model.criterion, optimumindex )
end

Distributions.rand!( model::MultiStartModel{T,U,V,F}, t::AbstractVector{T}, u::AbstractVector{U} ) where {T,U,V,F} =
    rand!( model.models[model.optimumindex], t, u )

Dependencies.compress( model::MultiStartModel{T,U,V,F} ) where {T,U,V,F} = Dependencies.compress.( model.models )

# assume they're all the same
initializationcount( model::Type{MultiStartModel{T,U,V,F}} ) where {T,U,V,F} = initializationcount( V )

function initialize( model::MultiStartModel{T,U,V,F}, t::AbstractVector{T}, u::AbstractVector{U} ) where {T,U,V,F}
    if model.optimumindex == 0
        # if we haven't yet worked out which model is optimal, we need to initialize them all
        for submodel in model.models
            initialize( submodel, t, u )
        end
    else
        initialize( model.models[model.optimumindex], t, u )
    end
end

state( model::MultiStartModel{T,U,V,F} ) where {T,U,V,F} = state( model.models[model.optimumindex] )

rootmodel( model::MultiStartModel{T,U,V,F} ) where {T,U,V,F} = rootmodel( model.models[model.optimumindex] )

Base.rand( model::MultiStartModel{T,U,V,F} ) where {T,U,V,F} = rand( model.models[model.optimumindex] )

mutable struct AdaptedModel{T, U, V <: AbstractModel{T,U}} <: AbstractModel{T,U}
    modelperiods::AbstractVector{T}
    models::Vector{V}
    lastperiod::Vector{T}
end

function rand( ::Type{AdaptedModel{T,U,V}}; modelperiods::AbstractVector{T} = T[], kwargs... ) where {T,U,V}
    model = rand( V; kwargs... )
    return AdaptedModel( modelperiods, [model], T[] )
end

function update( model::AdaptedModel{T,U,V}, t::T, u::U; kwargs... ) where {T,U,V}
    update( model.models[end], t, u )

    index = length(model.models)
    if model.lastperiod[1] < model.modelperiods[index] <= t
        println( "Fitting current model at $t" )
        model.models[end] = fit( model.models[end]; kwargs... )
    end

    model.lastperiod[1] = t

    if index < length(model.modelperiods) && t >= model.modelperiods[index+1]
        @assert( length(model.models) == index )
        push!( model.models, deepcopy( model.models[end] ) )
        println( "Fitting next model at $t" )
        model.models[end] = fit( model.models[end]; kwargs... )
    end
end

function Distributions.rand!( model::AdaptedModel{T,U,V}, t::AbstractVector{T}, u::AbstractVector{U} ) where {T,U,V}
    index = searchsorted( model.modelperiods, model.lastperiod[1] ).stop
    return rand!( model.models[index], t, u )
end

Dependencies.compress( model::AdaptedModel{T,U,V} ) where {T,U,V} = Dependencies.compress.( model.models )

initializationcount( model::Type{AdaptedModel{T,U,V}} ) where {T,U,V} = max( initializationcount( V ), 1 )

function initialize( model::AdaptedModel{T,U,V}, t::AbstractVector{T}, args... ) where {T,U,V}
    model.models = [model.models[1]]
    initialize( model.models[end], t, args... )
    model.lastperiod = t[end:end]
end

function state( model::AdaptedModel{T,U,V} ) where {T,U,V}
    index = searchsorted( model.modelperiods, model.lastperiod[1] ).stop
    return state( model.models[index] )
end

function rootmodel( model::AdaptedModel{T,U,V} ) where {T,U,V}
    index = searchsorted( model.modelperiods, model.lastperiod[1] ).stop
    return rootmodel( model.models[index] )
end

function Base.rand( model::AdaptedModel{T,U,V} ) where {T,U,V}
    index = searchsorted( model.modelperiods, model.lastperiod[1] ).stop
    return Base.rand( model.models[index] )
end

mutable struct RewindableModel{T, U, V <: AbstractModel{T,U}} <: AbstractModel{T,U}
    model::V
    t::Vector{T}
    u::Vector{U}
end

rand( ::Type{RewindableModel{T, U, V}}; kwargs... ) where {T,U,V} = RewindableModel( rand( V; kwargs... ), T[], U[] )

function update( model::RewindableModel{T,U,V}, t::T, u::U; kwargs... ) where {T,U,V}
    update( model.model, t, u; kwargs... )
    push!( model.t, t )
    push!( model.u, u )
end

Distributions.rand!( model::RewindableModel{T,U,V}, t::AbstractVector{T}, u::AbstractVector{U} ) where {T,U,V} =
    rand!( model.model, t, u )

Dependencies.compress( model::RewindableModel{T,U,V} ) where {T,U,V} = Dependencies.compress( model.model )

initializationcount( model::Type{RewindableModel{T,U,V}} ) where {T,U,V} = initializationcount( V )

initialize( model::RewindableModel{T,U,V}, args... ) where {T,U,V} = initialize( model.model, args... )

state( model::RewindableModel{T,U,V} ) where {T,U,V} = state( model.model )

rootmodel( model::RewindableModel{T,U,V} ) where {T,U,V} = rootmodel( model.model )

Base.rand( model::RewindableModel{T,U,V} ) where {T,U,V} = rand( model.model )

function reupdate( model::RewindableModel{T,U,V}; kwargs... ) where {T,U,V}
    n = initializationcount( V )
    initialize( model.model, model.t[1:n], model.u[1:n] )
    update( model.model, model.t[n+1:end], model.u[n+1:end]; kwargs... )
end

fit( model::RewindableModel{T,U,V}; kwargs... ) where {T,U,V} =
    RewindableModel( fit( model.model; kwargs... ), model.t, model.u )

mutable struct ANModel{T, U, V <: AbstractModel{T,U}} <: AbstractModel{T,U}
    rootmodel::V
    models::Vector{V}
    index::Int
    distribution::Vector{Distribution}
end

rand( ::Type{ANModel{T,U,V}}; kwargs... ) where {T,U,V} = ANModel{T,U,V}( rand( V; kwargs... ), V[], 1, Distribution[] )

function update( model::ANModel{T,U,V}, t::T, u::U; kwargs... ) where {T,U,V}
    update( model.rootmodel, t, u; kwargs... )
    for submodel in model.models
        update( submodel, t, u; kwargs... )
    end
    model.index = 1
end

function Base.rand( model::ANModel{T,U,V} ) where {T,U,V}
    if model.index > length(model.models)
        if isempty(model.distribution)
            C = sandwich( model )
            model.distribution = [MvNormal( zeros(size(C,1)), C )]
        end
        newmodel = deepcopy( model.rootmodel )
        setcompressedparameters!( newmodel, getcompressedparameters( newmodel ) + Base.rand( model.distribution[1] ) )
        reupdate( newmodel )
        push!( model.models, newmodel )
    end
    model.index += 1
    return model.models[model.index-1]
end

initializationcount( model::Type{ANModel{T,U,V}} ) where {T,U,V} = initializationcount( V )

initialize( model::ANModel{T,U,V}, args... ) where {T,U,V} = initialize( model.rootmodel, args... )

state( model::ANModel{T,U,V} ) where {T,U,V} = state( model.rootmodel )

rootmodel( model::ANModel{T,U,V} ) where {T,U,V} = rootmodel( model.rootmodel )

fit( model::ANModel{T,U,V}; kwargs... ) where {T,U,V} = ANModel( fit( model.rootmodel; kwargs... ), V[], 0, Distribution[] )

Distributions.rand!( model::ANModel{T,U,V}, t::AbstractVector{T}, u::AbstractVector{U} ) where {T,U,V} =
    rand!( model.rootmodel, t, u )

end # module
