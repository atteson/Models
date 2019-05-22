module Models

using Dates
using Distributions
using Random
using Distributed

abstract type AbstractModel{T}
end

initialize( M::AbstractModel ) = error( "initialize not yet implemented for $(typeof(M))" )

update( M::AbstractModel{T}, y::T ) where {T} = error( "update not yet implemented for $(typeof(M))" )

function update( M::AbstractModel{T}, a::AbstractVector{T} ) where {T}
    for x in a
        update( M, x )
    end
end

Distributions.rand!( M::AbstractModel{T}, v::AbstractVector{T}, n::Int = length(v) ) where {T} =
    error( "rand not yet implemented for $(typeof(M))" )

function Base.rand( M::AbstractModel, n::Int; kwargs... )
    observations = zeros(n)
    Distributions.rand!( M, observations; kwargs... )
    return observations
end

Base.rand( ::Type{U} ) where {T,U <: AbstractModel{T}} = error( "rand not implemented for type $U" )

fit( M::AbstractModel ) = error( "fit not yet implemented for $(typeof(M))" )

struct FittableModel{T, U <: AbstractModel{T},F <: Function} <: AbstractModel{T}
    model::U
end

FittableModel( model::U, ::F ) where {T,U <: AbstractModel{T}, F} = FittableModel{T,U,F}( model )

function fit( model::FittableModel{T,U,F}; kwargs... ) where {T,U,F}
    F.instance( model.model; kwargs... )
    return model
end

update( model::FittableModel{T}, y::T ) where {T} = update( model.model, y )

Base.rand( ::Type{FittableModel{T,U,F}}; kwargs... ) where {T, U, F} = FittableModel( rand( U; kwargs... ), F.instance )

abstract type DatedModel{T} <: AbstractModel{Tuple{Date,T}}
end

update( model::DatedModel{T}, y::Tuple{Date,T} ) where {T} = error( "update not yet implemented for $(typeof(M))" )
    
date( M::DatedModel ) = error( "date not yet implemented for $(typeof(M))" )

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

function Distributions.rand!( model::LogReturnModel, v::AbstractVector{Float64}, n::Int = length(v) ) where {T}
    rand!( model.model, v, n=n )
    lastprice = model.lastprice
    for i = 1:n
        v[i] = lastprice *= exp( v[i] )
    end
end

Base.rand( ::Type{LogReturnModel{T}}; lastdate::Date = nothing, lastprice::Float64 = nothing, kwargs... ) where {T} =
    LogReturnModel( rand( T; kwargs... ), lastdate, lastprice )

function fit( model::LogReturnModel; kwargs... )
    fit( model.model; kwargs... )
    return model
end

date( model::LogReturnModel ) = model.lastdate

mutable struct MultiStartModel{T,U <: AbstractModel{T}} <: AbstractModel{T}
    models::Vector{U}
end

function Base.rand(
    ::Type{MultiStartModel{U}};
    seeds::AbstractVector{Int} = 1:1,
    kwargs...
) where {T, U <: AbstractModel{T}}
    models = U[]
    for seed in seeds
        Random.seed!( seed )
        push!( models, rand( U; kwargs... ) )
    end
    return MultiStartModel( models )
end

function update( model::MultiStartModel{T,U}, y::T ) where {T,U}
    for submodel in model.models
        update( submodel, y )
    end
end

function fit(
    model::MultiStartModel;
    processes::Int = 1,
    modules::Vector{Symbol} = Symbol[],
    kwargs...
)
    if processes == 1
        for submodel in model.models
            fit( submodel; kwargs... )
        end
    else
        procs = addprocs(processes)
        
        # I don't know of a more convenient way to load all the modules we want
        futures = Future[]
        for pid in procs
            for moduletoeval in modules
                push!( futures, remotecall( Core.eval, pid, Main, Expr(:using,Expr(:.,moduletoeval)) ) )
            end
        end
        for future in futures
            wait(future)
        end

        model.models = pmap( submodel -> fit( submodel; kwargs... ), WorkerPool(procs), model.models )

        rmprocs(procs)
    end
    return model
end

mutable struct AdaptedModel{T,U <: AbstractModel{T}} <: DatedModel{T}
    dates::Vector{Date}
    models::Vector{U}
    index::Int
    lastprice::Float64
end

function AdaptedModel{T,U}( modeldir::String, d::Date ) where {T,U}
    fileregex = r"_([0-9]{8})$"
    dateformat = Dates.DateFormat( "yyyymmdd" )
    dates = Date[]
    models = T[]
    for file in readdir(modeldir)
        m = match( fileregex, file )
        if m == nothing
            error( "Couldn't match $file" )
        end
        push!( dates, Date( m.captures[1], dateformat ) )

        open( joinpath( modeldir, file ), "r") do f
            push!( models, read( f, T ) )
        end
        for model in models
            initialize( model )
        end
    end

    index = searchsorted( dates, d ).stop
    if index == 0
        error( "No model available at date $d" )
    end
    return AdaptedModel( dates, models, index )
end    

function update( model::AdaptedModel, d::Date, y::T ) where {T}
    while model.index < length(model.dates) && model.dates[model.index+1] < d
        model.index += 1
    end
    update( model.models[model.index], y )
end

Distributions.rand!( model::AdaptedModel, v::AbstractVector{Float64}, n::Int = length(v) ) =
    Distributions.rand!( model.models[model.index], v, n )

end # module
