using Models
using HMMs
using Brobdingnag
using Dates
using Distributions
using Random

Random.seed!(1)
hmm1 = rand( HMMs.HMM{2,Normal,Brob,Float64} )
y = rand( hmm1, 1_000 );

r = 0.01*(y .- mean(y))./std(y) .+ 0.0005

Random.seed!(2)
hmm2 = Models.FittableModel( rand( HMMs.HMM{2,Normal,Brob,Float64} ), HMMs.em )
Models.update( hmm2, r )
Models.fit( hmm2, debug=2 )

p = exp.(cumsum(r))

Random.seed!(2)
firstdate = Date(2004, 7, 2)
hmm3 = Models.LogReturnModel( Models.FittableModel( rand( HMMs.HMM{2,Normal,Brob,Float64} ), HMMs.em ), firstdate, 1.0 )

Models.update( hmm3, collect(zip(firstdate:Day(1):firstdate+Day(length(p)-1), p)) )
Models.fit( hmm3, debug=2 )

@assert( maximum( abs.(hmm2.model.initialprobabilities - hmm3.model.model.initialprobabilities) ) < 1e-8 )
@assert( maximum( abs.(hmm2.model.transitionprobabilities - hmm3.model.model.transitionprobabilities) ) < 1e-8 )
@assert( maximum( abs.(hmm2.model.stateparameters - hmm3.model.model.stateparameters) ) < 1e-8 )

modeltype = Models.MultiStartModel{Models.FittableModel{Float64,HMMs.HMM{2,Normal,Brob,Float64}, typeof(HMMs.em)}}
model = rand( modeltype, seeds=1:5 )
Models.update( model, r )
Models.fit( model, debug=2 )

