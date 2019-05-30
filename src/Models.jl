module Models

using Dates
using Distributions
using Random
using Distributed
using Dependencies

abstract type AbstractModel{T}
end

function update( M::AbstractModel{T}, a::AbstractVector{T}; kwargs... ) where {T}
    for x in a
        update( M, x; kwargs... )
    end
end

function Base.rand( M::AbstractModel, n::Int; kwargs... )
    observations = zeros(n)
    Distributions.rand!( M, observations; kwargs... )
    return observations
end

struct FittableModel{T, U <: AbstractModel{T}, F <: Function} <: AbstractModel{T}
    model::U
    f::F
end

fit( model::FittableModel{T,U,F}; kwargs... ) where {T,U,F} = FittableModel( model.f( model.model; kwargs... ), model.f )

update( model::FittableModel{T}, y::T ) where {T} = update( model.model, y )

Dependencies.getinstance( ::Type{F} ) where {F <: Function} = F.instance

Base.rand( ::Type{FittableModel{T,U,F}}; fitfunction::F = Dependencies.getinstance( F ), kwargs... ) where {T, U, F} =
    FittableModel( rand( U; kwargs... ), fitfunction )

Distributions.rand!( model::FittableModel{T,U,F}, v::AbstractVector{T}, n::Int = length(v) ) where {T,U,F} =
    rand!( model.model, v, n )

Dependencies.compress( model::FittableModel{T,U,F} ) where {T,U,F} = Dependencies.compress( model.model )

abstract type DatedModel{T} <: AbstractModel{Tuple{Date,T}}
end

mutable struct LogReturnModel{T <: AbstractModel{Float64}} <: DatedModel{Float64}
    model::T
    lastdate::Date
    lastprice::Float64
end

initialize( model::LogReturnModel ) = initialize( model.model )

function update( model::LogReturnModel, y::Tuple{Date,Float64} )
    update( model.model, log(y[2]/model.lastprice) )
    model.lastdate = y[1]
    model.lastprice = y[2]
end

function Distributions.rand!( model::LogReturnModel{T}, v::AbstractVector{Float64}, n::Int = length(v) ) where {T}
    rand!( model.model, v, n )
    lastprice = model.lastprice
    for i = 1:n
        v[i] = lastprice *= exp( v[i] )
    end
end

Base.rand( ::Type{LogReturnModel{T}}; lastdate::Date = nothing, lastprice::Float64 = nothing, kwargs... ) where {T} =
    LogReturnModel( rand( T; kwargs... ), lastdate, lastprice )

fit( model::LogReturnModel; kwargs... ) = LogReturnModel( fit( model.model; kwargs... ), model.lastdate, model.lastprice )

date( model::LogReturnModel ) = model.lastdate

Dependencies.compress( model::LogReturnModel{T} ) where {T} = Dependencies.compress( model.model )

mutable struct MultiStartModel{T, U <: AbstractModel{T}, F <: Function} <: AbstractModel{T}
    models::Vector{U}
    criterion::F
    optimumindex::Int
end

function Base.rand(
    ::Type{MultiStartModel{T,U,F}};
    seeds::AbstractVector{Int} = 1:1,
    kwargs...
) where {T, U <: AbstractModel{T}, F}
    models = U[]
    for seed in seeds
        Random.seed!( seed )
        push!( models, rand( U; kwargs... ) )
    end
    return MultiStartModel( models, Dependencies.getinstance( F ), 0 )
end

function update( model::MultiStartModel{T,U,F}, y::T ) where {T,U,F}
    for submodel in model.models
        update( submodel, y )
    end
end

function fit(
    model::MultiStartModel{T,U,F};
    modules::Vector{Symbol} = Symbol[],
    kwargs...
) where {T,U,F}
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

Distributions.rand!( model::MultiStartModel{T,U,F}, v::AbstractVector{Float64}, n::Int = length(v) ) where {T,U,F} =
    rand!( model.models[model.optimumindex], v, n )

Dependencies.compress( model::MultiStartModel{T,U,F} ) where {T,U,F} = Dependencies.compress.( model.models )

mutable struct AdaptedModel{T,U <: DatedModel{T}} <: DatedModel{T}
    modeldates::AbstractVector{Date}
    models::Vector{U}
    lastdate::Date
end

date( model::AdaptedModel{T,U} ) where {T,U} = model.lastdate

function Base.rand( ::Type{AdaptedModel{T,U}}; modeldates::AbstractVector{Date} = Date[], kwargs... ) where {T,U}
    model = rand( U; kwargs... )
    return AdaptedModel( modeldates, [model], Date(0) )
end

function update( model::AdaptedModel{T,U}, y::Tuple{Date, T}; kwargs... ) where {T,U}
    @assert( date( model ) < y[1] )
    update( model.models[end], y )

    index = length(model.models)
    if model.lastdate < model.modeldates[index] <= y[1]
        println( "Fitting current model at $(y[1])" )
        model.models[end] = fit( model.models[end]; kwargs... )
    end

    model.lastdate = y[1]

    if index < length(model.modeldates) && y[1] >= model.modeldates[index+1]
        @assert( length(model.models) == index )
        push!( model.models, deepcopy( model.models[end] ) )
        println( "Fitting next model at $(y[1])" )
        model.models[end] = fit( model.models[end]; kwargs... )
    end
end

function Distributions.rand!( model::AdaptedModel{T,U}, v::AbstractVector{T}, n::Int = length(v) ) where {T,U}
    index = searchsorted( model.modeldates, model.lastdate ).stop
    return rand!( model.models[index], v, n )
end

Dependencies.compress( model::AdaptedModel{T,U} ) where {T,U} = Dependencies.compress.( model.models )

end # module
