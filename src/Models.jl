module Models

using Dates

abstract type AbstractModel
end

initialize( M::AbstractModel ) = error( "initialize not yet implemented for $(typeof(M))" )

roll( M::AbstractModel ) = error( "roll not yet implemented for $(typeof(M))" )

update( M::AbstractModel, y::Float64 ) = error( "update not yet implemented for $(typeof(M))" )

rand( M::AbstractModel, n::Int ) = error( "rand not yet implemented for $(typeof(M))" )

mutable struct AdaptedModel{T <: AbstractModel}
    dates::Vector{Date}
    models::Vector{T}
    index::Int
end

function AdaptedModel{T}( modeldir::String ) where {T}
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

    return AdaptedModel( dates, models, 0 )
end    

function update( M::AdaptedModel{T}, d::Date, y::Float64 ) where {T}
    if M.index == 0
        M.index = searchsorted( M.dates, d ).stop
        if M.index == 0
            error( "No model available at date $d" )
        end
    else
        while M.index < length(M.dates) && M.dates[M.index+1] < d
            M.index += 1
        end
        update( M.models[M.index], y )
    end
end

end # module
