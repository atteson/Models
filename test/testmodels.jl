using Models
using HMMs
using Brobdingnag
using Dates
using Distributions
using Random

Random.seed!(1)
hmm1 = rand( HMMs.HMM{2,Normal,Brob,Float64} )
y = rand( hmm1, 1_000 );

Random.seed!(2)
hmm2 = Models.FittableModel( rand( HMMs.HMM{2,Normal,Brob,Float64} ), HMMs.em )
Models.update( hmm2, y )
Models.fit( hmm2, debug=2 )
