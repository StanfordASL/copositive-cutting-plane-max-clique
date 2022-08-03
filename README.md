# copositive-cutting-plane-max-clique

This code base is using the Julia Language and [DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project named
> copositive-cutting-plane-max-clique

To (locally) reproduce this project, do the following:

0. Download this code base. Notice that raw data are typically not included in the
   git-history and may need to be downloaded independently.
   
1. Set up your conda environment to use dwave-neal:
	```
	> conda create --name "myenv" python=3.9
	> conda activate myenv
	> python -m pip install -r requirements.txt
	```
	
2. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
   julia> Pkg.activate("path/to/this/project/src/COP")
   julia> Pkg.build("Gurobi") # you may need to set ENV["GUROBI_HOME"] prior if it isn't set
   julia> Pkg.activate("path/to/this/project/")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the julia scripts and
everything should work out of the box.

3. To run the copositive cutting plane algorithm:
	```
	> cd path/to/this/project/scripts
	> julia run_cutting_plane.jl exp_raw --param {#} --anst --neal
	```
    Replace {#} with 25, 05, or 75  
    --anst runs the cutting plane algorithm with Anstreicher's MILP copositivity formulation  
    --neal runs the cutting plane algorithm with dwave neal as the copositivity checker  

4. To run additional benchmarks:
	```
	> python benchmarking.py exp_raw --param {#} --hpo --fixed --gp_cop --gp_mip --dw_mc
	```
    Replace {#} with 25, 05, or 75  
    --hpo run hyperopt parameter tuning'  
    --fixed run fixed neal with 100 sweeps  
    --gp_cop run gurobi copositivity check experiment  
    --gp_mip run gurobi mip formulations  
    --dw_mc run dwave maximum clique sampler  
    
    Note that the graphs are generated in 2. so it needs to be run before 4. 

5. To generate plots: run cells in notebooks/plotting.ipynb
	
