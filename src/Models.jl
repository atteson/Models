module Models

mutable struct Model{T}
    dates::Vector{Date}
    models::Vector{T}
end

function Model{T}( modeldir::String )
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

        open( file, "r") do f
            push!( models, read( f, T ) )
        end
        for model in models
            initialize( model )
        end
    end

    return Model( dates, models )
end    

end # module
