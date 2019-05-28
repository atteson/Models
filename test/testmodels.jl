using Models
using HMMs
using Brobdingnag
using Dates
using Distributions
using Random
using Distributed

hmm1 = HMMs.HMM{2,Normal,Brob,Float64}( [0.5, 0.5], [0.9 0.1;0.05 0.95], [-0.001 0.0005;0.02 0.01] )
Random.seed!(1)
y = rand( hmm1, 1_000 );

hmm2 = Models.FittableModel( rand( HMMs.HMM{2,Normal,Brob,Float64}, seed=2 ), HMMs.em )
Models.update( hmm2, y )
Models.fit( hmm2, debug=2 )

relerr( x, y ) = abs(x - y)/(abs(x) + 1)

HMMs.reorder!( hmm1 )
HMMs.reorder!( hmm2.model )
@assert( maximum(relerr.(hmm2.model.transitionprobabilities, hmm1.transitionprobabilities)) < 0.05 )
@assert( maximum(relerr.(hmm2.model.stateparameters, hmm1.stateparameters)) < 0.05 )

p = exp.(cumsum(y))

firstdate = Date(2004, 7, 2)
hmm3 = Models.LogReturnModel( Models.FittableModel( rand( HMMs.HMM{2,Normal,Brob,Float64}, seed=2 ), HMMs.em ), firstdate, 1.0 )

Models.update( hmm3, collect(zip(firstdate:Day(1):firstdate+Day(length(p)-1), p)) )
Models.fit( hmm3, debug=2 )

hmmerr( hmm1, hmm2 ) = 
    max( maximum( abs.(hmm1.initialprobabilities - hmm2.initialprobabilities) ),
         maximum( abs.(hmm1.transitionprobabilities - hmm2.transitionprobabilities) ),
         maximum( abs.(hmm1.stateparameters - hmm2.stateparameters) ) )

@assert( hmmerr( hmm2.model, hmm3.model.model ) .< 1e-8 )

models = [Models.FittableModel( rand( HMMs.HMM{2,Normal,Brob,Float64}, seed=seed ), HMMs.em ) for seed in 1:5]
hmm4 = Models.MultiStartModel( models )
Models.update( hmm4, y )
@time Models.fit( hmm4, debug=2 );

myworkers = addprocs(2)

models = [Models.FittableModel( rand( HMMs.HMM{2,Normal,Brob,Float64}, seed=seed ), HMMs.em ) for seed in 1:5]
hmm5 = Models.MultiStartModel( models )
Models.update( hmm5, y )
Models.fit( hmm5, debug=2, modules=[:HMMs,:Brobdingnag] )

errs = [hmmerr( hmm4.models[i].model, hmm5.models[i].model ) for i in 1:max(length(hmm4.models),length(hmm5.models))]
@assert( all(isnan.(errs) .| (errs .== 0.0)) )

rmprocs(myworkers)

hmmtype = HMMs.HMM{2,Normal,Brob,Float64}
hmm6 = rand( hmmtype, seed=1 )
y = rand( hmm6, 10_000 )
dates = map( i -> Date(1985,1,1) + Day(i), 1:length(y) )

modeldates = dates[5_000]:Year(1):dates[10_000]
hmm7 = Models.AdaptedModel( modeldates = modeldates, rand( hmmtype, seed=2 ) )
Models.update( hmm7, collect(zip(dates,y))[1:6_000], debug=2 )

model.models

[findall(dates.==modeldate)[1] for modeldate in modeldates]
[length(submodel.model.y) for submodel in model.models]
