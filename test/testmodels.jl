using Models
using HMMs
using Brobdingnag
using Dates
using Distributions
using Random
using Distributed
using Dependencies

hmm1 = HMMs.HMM{2,Normal,Brob,Float64}( [0.5, 0.5], [0.9 0.1;0.05 0.95], [-0.001 0.0005;0.02 0.01] )
Random.seed!(1)
y = rand( hmm1, 1_000 );

Random.seed!(2)
hmm2 = Models.FittableModel( rand( HMMs.HMM{2,Normal,Brob,Float64} ), HMMs.em )
Models.update( hmm2, y )
hmm2 = Models.fit( hmm2, debug=2 )

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

hmmtype = Models.FittableModel{Float64, HMMs.HMM{2,Normal,Brob,Float64}, typeof(HMMs.em)}
criterion = model -> HMMs.likelihood( model.model )
modeltype = Models.MultiStartModel{Float64, hmmtype, typeof(criterion)}
hmm4 = rand( modeltype, seeds=1:5, fitfunction = HMMs.em )
Models.update( hmm4, y )
@time Models.fit( hmm4, debug=2 );

myworkers = addprocs(2)

hmm5 = rand( modeltype, seeds=1:5 )
Models.update( hmm5, y )
Models.fit( hmm5, debug=2, modules=[:HMMs,:Brobdingnag] )

errs = [hmmerr( hmm4.models[i].model, hmm5.models[i].model ) for i in 1:max(length(hmm4.models),length(hmm5.models))]
@assert( all(isnan.(errs) .| (errs .== 0.0)) )

rmprocs(myworkers)

hmm6 = rand( modeltype, seeds=1:50, fitfunction = HMMs.em )
Models.update( hmm6, y )
@time Models.fit( hmm6, debug=2 );
# 4.6s

fittablemodeltype = Models.FittableModel{Float64, modeltype, FunctionNode{typeof(Models.fit)}}
hmm7 = rand( fittablemodeltype, seeds=1:50 )
Models.update( hmm7, y )
@time Models.fit( hmm7, debug=2 );
# 5.1s

hmm8 = rand( fittablemodeltype, seeds=1:50 )
Models.update( hmm8, y )
@time Models.fit( hmm8, debug=2 );
# 0.4s

hmm9 = rand( fittablemodeltype, seeds=1:50 )
Models.update( hmm9, y )
delete!( hmm7.f, hmm9.model, debug=2 )

Random.seed!(1)
hmm10 = rand( hmmtype )
y = rand( hmm10, 10_000 )
dates = map( i -> Date(1985,1,1) + Day(i), 1:length(y) )

modeldates = dates[5_000]:Year(1):dates[10_000]

modeltype = Models.AdaptedModel{Float64,Models.LogReturnModel{hmmtype}}
model = rand( modeltype, modeldates=modeldates, lastdate=dates[1], lastprice=exp(y[1]) )

Models.update( model, collect(zip(dates,exp.(y)))[2:6_000], debug=2 )

model.models

[findall(dates.==modeldate)[1] for modeldate in modeldates]
[length(submodel.model.model.y) for submodel in model.models]
