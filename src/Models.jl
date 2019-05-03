module Models

using Dates
using Distributions

abstract type AbstractModel
end

initialize( M::AbstractModel ) = error( "initialize not yet implemented for $(typeof(M))" )

roll( M::AbstractModel ) = error( "roll not yet implemented for $(typeof(M))" )

update( M::AbstractModel, y::Float64 ) = error( "update not yet implemented for $(typeof(M))" )

Distributions.rand!( M::AbstractModel, v::AbstractVector, n::Int = length(v) ) =
    error( "rand not yet implemented for $(typeof(M))" )

function Base.rand( M::AbstractModel, n::Int )
    observations = zeros(n)
    Distributions.rand!( M, observations )
    return observations
end

mutable struct AdaptedModel{T <: AbstractModel}
    dates::Vector{Date}
    models::Vector{T}
    index::Int
end

function AdaptedModel{T}( modeldir::String, d::Date ) where {T}
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

function update( model::AdaptedModel, d::Date, y::Float64 )
    while model.index < length(model.dates) && model.dates[model.index+1] < d
        model.index += 1
    end
    update( model.models[model.index], y )
end

Distributions.rand!( model::AdaptedModel, v::AbstractVector{Float64}, n::Int = length(v) ) =
    Distributions.rand!( model.models[model.index], v, n )

Base.rand( model::AdaptedModel, n::Int ) = rand( model.models[model.index], n )

end # module
