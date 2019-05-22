using Models
using HMMs
using Brobdingnag
using Dates
using Distributions
using Random

hmm1 = HMMs.HMM{2,Normal,Brob,Float64}( [0.5, 0.5], [0.9 0.1;0.05 0.95], [-0.001 0.0005;0.02 0.01] )
Random.seed!(1)
y = rand( hmm1, 1_000 );

Random.seed!(2)
hmm2 = Models.FittableModel( rand( HMMs.HMM{2,Normal,Brob,Float64} ), HMMs.em )
Models.update( hmm2, y )
Models.fit( hmm2, debug=2 )

relerr( x, y ) = abs(x - y)/(abs(x) + 1)

HMMs.reorder!( hmm1 )
HMMs.reorder!( hmm2.model )
@assert( maximum(relerr.(hmm2.model.transitionprobabilities, hmm1.transitionprobabilities)) < 0.05 )
@assert( maximum(relerr.(hmm2.model.stateparameters, hmm1.stateparameters)) < 0.05 )

p = exp.(cumsum(y))

Random.seed!(2)
firstdate = Date(2004, 7, 2)
hmm3 = Models.LogReturnModel( Models.FittableModel( rand( HMMs.HMM{2,Normal,Brob,Float64} ), HMMs.em ), firstdate, 1.0 )

Models.update( hmm3, collect(zip(firstdate:Day(1):firstdate+Day(length(p)-1), p)) )
Models.fit( hmm3, debug=2 )

hmmerr( hmm1, hmm2 ) = 
    max( maximum( abs.(hmm1.initialprobabilities - hmm2.initialprobabilities) ),
         maximum( abs.(hmm1.transitionprobabilities - hmm2.transitionprobabilities) ),
         maximum( abs.(hmm1.stateparameters - hmm2.stateparameters) ) )

@assert( hmmerr( hmm2.model, hmm3.model.model ) .< 1e-8 )

modeltype = Models.MultiStartModel{Models.FittableModel{Float64,HMMs.HMM{2,Normal,Brob,Float64}, typeof(HMMs.em)}}
model1 = rand( modeltype, seeds=1:5 )
Models.update( model1, y )
Models.fit( model1, debug=2 )

model2 = rand( modeltype, seeds=1:5 )
Models.update( model2, y )
Models.fit( model2, debug=2, processes=2, modules=[:HMMs,:Brobdingnag] )

errs = [hmmerr( model1.models[i].model, model2.models[i].model ) for i in 1:max(length(model1.models),length(model2.models))]
@assert( all(isnan.(errs) .| (errs .== 0.0)) )



