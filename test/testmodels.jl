using Models
using HMMs
using Brobdingnag

modeldir = joinpath( dirname(dirname(pathof(Models))), "data" )

models = Models.AdaptedModel{HMMs.HMM{HMMs.GenTDist,Brob,Float64}}( modeldir ) )
