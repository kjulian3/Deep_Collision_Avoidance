module PilotSCAViz_DRL

export viz_policy_drl

import PilotSCAConst_DRL: RangeMin, RangeMax, ThetaMin, ThetaMax, PsiMin, PsiMax
import PilotSCAConst_DRL: SpeedOwnMin, SpeedOwnMax, SpeedIntMin, SpeedIntMax
import PilotSCAConst_DRL: ThetaDim, PsiDim, SpeedOwnDim, SpeedIntDim
import PilotSCAConst_DRL: Thetas, Psis, SpeedInts, SpeedOwns, Actions


using Interact, PGFPlots, Colors, ColorBrewer


const STATE_DIM = 5
const ACTION_DIM = 6

const R  = 1   #Range    
const Th = 2   #Theta            
const Ps = 3   #Psi             
const So = 4   #SpeedOwn        
const Si = 5   #SpeedInt            

const RANGEMAX = RangeMax
const RANGEMIN = RangeMin


const THDIM = ThetaDim
const PSDIM = PsiDim
const SODIM = SpeedOwnDim
const SIDIM = SpeedIntDim

const thetas = Thetas
const psis   = Psis
const sos    = SpeedOwns
const sis    = SpeedInts
const libpathBlas = Pkg.dir("..", "..", "kyle", "library")
global const LIB_BLAS = Libdl.find_library(collect(["libnnet_blas"]), collect([libpathBlas]))

const libpath = Pkg.dir("..", "..", "kyle", "library")
global const LIB = Libdl.find_library(collect(["libnnet"]), collect([libpath]))

function viz_policy_drl(libraryPath::AbstractString, neuralNetworkPath::AbstractString, nearest::Bool=false,batch_size=1)
    
    #const LIB_BLAS = Libdl.find_library(collect(["libnnet_blas"]), collect([libraryPath]))
    pstart = round(rad2deg(psis[1]),0)
    pend   = round(rad2deg(psis[end]),0)
    pdiv   = round(rad2deg(psis[2]-psis[1]),0)
    
    v0start = sos[1]
    v0end   = sos[end]
    v0div   = sos[2]-sos[1]
    
    v1start = sis[1]
    v1end   = sis[end]
    v1div   = sis[2] - sis[1]
    
    c = RGB{U8}(1.,1.,1.) # white
    e = RGB{U8}(.0,.0,.5) # pink
    a = RGB{U8}(.0,.600,.0) # green
    d = RGB{U8}(.5,.5,.5) # grey
    b = RGB{U8}(.7,.9,.0) # neon green
    f = RGB{U8}(0.94,1.0,.7) # neon green
    colors =[a; b; f; c; d; e]
    #mat = ccall((:load_network,"/home/sisl/kyle/library/libnnet"),Ptr{Void}, (Ptr{Uint8},),"/home/sisl/kyle/library/params_MSE_RMSPROP2.nnet")
   
    @manipulate for psi_int  = round(rad2deg(psis)),#pstart:pdiv:pend,
        v_own = sos,#v0start:v0div:v0end, #sos,
        v_int = sis,#v1start:v1div:v1end, #sis,
        zoom = [4, 3, 2, 1.5,1],
        nbin = [100,150,200,250]
            
        mat =  ccall((:load_network,LIB_BLAS),Ptr{Void},(Ptr{UInt8},),neuralNetworkPath)
            inputsNet= zeros(nbin*nbin,STATE_DIM)    
            ind = 1
            for i=linspace(round(Int,-1*RANGEMAX/zoom),round(Int,RANGEMAX/zoom),nbin)
                for j=linspace(round(Int,-1*RANGEMAX/zoom),round(Int,RANGEMAX/zoom),nbin)
                    r = sqrt(i^2+j^2)
                    th = atan2(j,i)
                    inputsNet[ind,:] = [r,th,deg2rad(psi_int),v_own,v_int];
                    ind = ind+1
                end
            end            
            
            q_nnet = zeros(nbin*nbin,ACTION_DIM);
            q_temp = zeros(ACTION_DIM*batch_size)
            ind = 1
            while ind+batch_size<nbin*nbin
                input3 = inputsNet[ind:(ind+batch_size-1),:];input3 = input3'[:];
            ccall((:evaluate_network_multiple,LIB_BLAS),Int32,(Ptr{Void},Ptr{Float64},Int32,Ptr{Float64}),mat,input3,batch_size,q_temp)
                q_nnet = [q_nnet[1:(ind-1),:];reshape(q_temp,ACTION_DIM,batch_size)';q_nnet[ind+batch_size:end,:]]
                ind=ind+batch_size
            end
            input3 = inputsNet[ind:end,:];input3 = input3'[:];
            n_left = nbin*nbin-ind+1
            q_temp = zeros(ACTION_DIM*n_left)
            ccall((:evaluate_network_multiple,LIB_BLAS),Int32,(Ptr{Void},Ptr{Float64},Int32,Ptr{Float64}),mat,input3,n_left,q_temp)
            q_nnet = [q_nnet[1:(ind-1),:];reshape(q_temp,ACTION_DIM,n_left)']

            #q_temp = [0.0,0.0,0.0,0.0,0.0]
            #q_nnet = zeros(length(ranges)*length(thetas),5)
            #i=1
            #for r = ranges
            #    for th = thetas
            #    input = [r,th,deg2rad(psi_int),v_own,v_int,tau,pasTrue[pa]]
            #        qtemp = [0.0;0.0;0.0;0.0;0.0]
            #        @eval ccall((:evaluate_network,$libraryPath),Int32,(Ptr{Void},Ptr{Float64},Ptr{Float64}),$mat,$input,$qtemp)
            #        q_nnet[i,:]=qtemp
            #        i=i+1
            #    end
            #end
            #@eval ccall((:destroy_network,$libraryPath),Void,(Ptr{Void},),$mat)
            
        #policy2 = read_policy(ACTIONS,q_nnet)
        
        ind = 1
        
        #function get_heat1(x::Float64, y::Float64)           
        #   r = sqrt(x^2+y^2)
        #   th = atan2(y,x)  
        #    
        #    action, _ = evaluate(policy, get_belief(
        #    [pa,tau,v_int,v_own,deg2rad(psi_int),th,r],grid,nearest))         
        #   return rad2deg(action[1])
        #end # function get_heat1
        
       function get_heat2(x::Float64, y::Float64)              
           r = sqrt(x^2+y^2)
           th = atan2(y,x)            
            action  = Actions[indmax(q_nnet[ind,:])]
            ind = ind+1
            return action
       end # function get_heat2
        
        g = GroupPlot(1, 1, groupStyle = "horizontal sep=3cm")
        #push!(g, Axis([
        #    Plots.Image(get_heat1, (round(Int,-1*RANGEMAX/zoom), round(Int,RANGEMAX/zoom)), 
        #               (round(Int,-1*RANGEMAX/zoom), round(Int,RANGEMAX/zoom)), 
        #               zmin = -3, zmax = 3,
        #               xbins = nbin, ybins = nbin,
        #    colormap = ColorMaps.RGBArray(colors), colorbar=false),
        #   Plots.Node(L">", 0, 0, style=string("font=\\Huge")),
        #   Plots.Node(L">", 53000/zoom, 53000/zoom, style=string("rotate=", psi_int, ",font=\\Huge"))
        #    ], width="10cm", height="10cm", xlabel="x (ft)", ylabel="y (ft)", title="Q Table action"))
        push!(g, Axis([
           Plots.Image(get_heat2, (round(Int,-1*RANGEMAX/zoom), round(Int,RANGEMAX/zoom)), 
                       (round(Int,-1*RANGEMAX/zoom), round(Int,RANGEMAX/zoom)), 
                       zmin = -20, zmax = 20,
                       xbins = nbin, ybins = nbin,
                       colormap = ColorMaps.RGBArray(colors)),
           Plots.Node(L">", 0, 0, style=string("font=\\Huge")),
           Plots.Node(L">", 2500/zoom, 2500/zoom, style=string("rotate=", psi_int, ",font=\\Huge"))
            ], width="10cm", height="10cm", xlabel="x (ft)", ylabel="y (ft)", title="Neural Net action"))
        ccall((:destroy_network,LIB_BLAS),Void,(Ptr{Void},),mat)
        g
    end # for p_int, v0, v1, pa, ta
end # function viz_pairwise_policy

end #module PilotSCAViz
