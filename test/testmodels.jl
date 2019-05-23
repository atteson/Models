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
hmm4 = rand( modeltype, seeds=1:5 )
Models.update( hmm4, y )
Models.fit( hmm4, debug=2 )

addprocs(2)

hmm5 = rand( modeltype, seeds=1:5 )
Models.update( hmm5, y )
Models.fit( hmm5, debug=2, modules=[:HMMs,:Brobdingnag] )

errs = [hmmerr( hmm4.models[i].model, hmm5.models[i].model ) for i in 1:max(length(hmm4.models),length(hmm5.models))]
@assert( all(isnan.(errs) .| (errs .== 0.0)) )


modeltype = HMMs.HMM{2,HMMs.GenTDist,Brob,Float64}
hmm6 = modeltype( [0.5, 0.5], [0.9 0.1;0.05 0.95], [-0.001 0.0005; 0.02 0.01; 5.0 7.0] )
y2 = rand( hmm6, 5_000 )

Random.seed!(2)
hmm7 = rand( Models.FittableModel{Float64, modeltype, typeof(HMMs.em)} )
Models.update( hmm7, y2 )
Models.fit( hmm7; debug=2 )

# first times starts up optimizer
Random.seed!(2)
hmm7 = rand( Models.FittableModel{Float64, modeltype, typeof(HMMs.em)} )
Models.update( hmm7, y2 )
t0 = time()
Models.fit( hmm7; debug=2 )
elapsed7 = time() - t0
# 38.9s

Random.seed!(2)
hmm8 = rand( Models.FittableModel{Float64, modeltype, typeof(HMMs.em)} )
Models.update( hmm8, y2[1:2500] )
t0 = time()
Models.fit( hmm8; debug=2 )
elapsed8_0 = time() - t0
# 2.4s

Models.update( hmm8, y2[2501:end] )
t0 = time()
Models.fit( hmm8; debug=2 )
elapsed8_1 = time() - t0
# 34.7s

Random.seed!(2)
hmm9 = rand( Models.FittableModel{Float64, modeltype, typeof(HMMs.em)} )
Models.update( hmm9, y2[1:2500] )
t0 = time()
Models.fit( hmm9; debug=2 )
elapsed9_0 = time() - t0
# 2.4s

Models.update( hmm9, y2[2501] )
t0 = time()
Models.fit( hmm9; debug=2 )
elapsed9_1 = time() - t0
# 29.0s

Models.update( hmm9, y2[2502] )
t0 = time()
Models.fit( hmm9; debug=2 )
elapsed9_2 = time() - t0
# 10.2s

