#include <time.h>
#include<fstream>
#include<stdlib.h> 
#include<stdio.h>
#include<iostream>
#include<vector>
#include<math.h>
#include<utility>//for pair<>
#include<time.h>//for srand(). time
using namespace std;

/////////////////////////GA PARAMETERS(To be changed)//////////////////////////////

int cities_number_global=127;
int population_size_global = 100;//even prefereably
int num_iterations_global = 10000000;
double fitness_scaling_global = (10000*cities_number_global);

bool is_initialise_population_random = 0;
int initialisation_method_to_use= 1;//1 for nearest neighobour initialisation //2 for randomised nearest neighobour initialisation
double randomisation_in_RNN = 0.5;//0.9 times, instead of nearest neighbour, a feasible random node will be chosen

bool is_crossover_enabled = 0;//for simulated annealing ,the 2-opt mutation operator is used
double CrossOverProbability = 0.8;
double T_current = 100;//this global parameter will keep getting updated as the simulated annealing process goes on
double T_rate_of_decrease = (T_current/num_iterations_global);
int num_times_simulated_annealing_iteration = ((cities_number_global*(cities_number_global-1))/100);

bool is_simulated_annealing = 1;
bool is_RBIBNNM_mutation = 0;
bool is_custom_mutation = 0;//our 2-opt mutation operator
int num_iterns_custom_mutation = 1000;//check 100 times for edges that can be bettered using 2-opt
double MutationProbability = 0.01/cities_number_global;//one percent chance that a city sequence changes in any way

bool is_Elitism = 0;//need this, otherwise the performance is very poor
int number_elites = 1;//keep one for now, otherwise code will break in MakeNextGen

///////////////////////////////////////////////////////////////////////////////////

typedef pair<int,int> iPair; 

double dist_2d(iPair p1, iPair p2){
	return sqrt(pow(p1.first-p2.first,2) + pow(p1.second-p2.second,2));
}

clock_t start_time;

class Graph
{
	public:
		Graph(vector<iPair> city);
		vector<iPair> cities_coords;
		int num_vertices;
		double** adj_mat;
		int generation_number;
		vector<vector<int> > best_chromosome_global_list;
		vector<vector<int> > best_chromosome_generation_list;
		vector<double> best_fitness_global_list;
		vector<double> best_fitness_generation_list;
		vector<double> best_total_distance_global_list;
		vector<double> best_total_distance_generation_list;
		int fittest_index;
		
		vector<int> best_chromosome;
		double best_fitness;
		double best_fitness_generation;
		double best_total_distance;
		double best_total_distance_generation;
		
		vector<vector<int> > population;
		vector<double> fitness;
		vector<int> total_distance;
		int calculateTotalDistanceOfAChromosome(vector<int> duplicate_chromosome);
		void InitialisePopulation();
		void CalculateFitness();
		void DisplayBestSolution(int index);
		void MakeNextGen();
		void CrossOver();//position based crossover used
		void Mutation();
		void TSP();
};

Graph::Graph(vector<iPair> city)
{
	//cout<<"Setting graph variables"<<endl;
	int n = city.size();
	this->generation_number = 0;
	this->num_vertices = n;
	this->adj_mat = (double**)malloc(n*sizeof(double*));
	this->best_fitness=0;
	this->best_total_distance=0;
	//cout<<"HI"<<endl;
	for(int i=0;i<n;i++){
		(this->cities_coords).push_back(city[i]);
		(this->adj_mat)[i] = (double*)malloc(n*sizeof(double));
	}
	//cout<<"HI"<<endl;
	double dist_temp;
	for(int i=0;i<n;i++){
		for(int j=0;j<n;j++){
			//cout<<i<<","<<j<<endl;
			dist_temp = dist_2d(city[i], city[j]);
			(this->adj_mat)[i][j] = dist_temp;
		}
	}
	//cout<<"HI"<<endl;

	for(int i=0;i<population_size_global;i++){
		(this->total_distance).push_back(0);
		(this->fitness).push_back(0);
	}
	//cout<<"Set the graph variables"<<endl;
	this->best_fitness = -1111;
	for(int i=0; i<n; i++)
		(this->best_chromosome).push_back(i);
}

void swap(int &a, int &b){
	int temp=a; a=b; b=temp;
	return;
}

bool Iselem(int key, vector<int> arr){
	for(int i=0;i<arr.size();i++){
		if(arr[i] == key)
			return 1;
	}
	return 0;
}

int findNearestNeighbour(int node_index, vector<int> cities_order, vector<iPair> cities_coords){
	//cout<<"Entered findNearestNeighbour"<<endl;
	double dist_min = 1000000;
	int n = cities_coords.size();
	//cout<<"n: "<<n<<endl;
	int index_to_return = 0;
	for(int i=0;i<n;i++){
		if(Iselem(i, cities_order))
			continue;
		if(dist_2d(cities_coords[i], cities_coords[node_index]) < dist_min){
			index_to_return = i;
			dist_min = dist_2d(cities_coords[i], cities_coords[node_index]);
		}
	}	
	return index_to_return;
}

int findRandomNeighbour(int node_index, vector<int> cities_order, int n){
	//cout<<"Entered findRandomNeighbour"<<endl;
	int rand_k = rand()%n;
	for(int i=0;i<n;i++){
		if(Iselem(rand_k, cities_order)){
			rand_k++;
			if(rand_k==n)
				rand_k=0;
			continue;
		}
		return rand_k;
	}
	
	return 0;
}

void printChromosome(vector<int> arr)
{
	for(int i=0;i<arr.size();i++)
		cout<<arr[i]<<" ";
	cout<<endl;
}

void Graph::InitialisePopulation()
{
	int n = this->num_vertices;
	vector<int> cities_order;
	if(!is_initialise_population_random){
		//nearest neighbour initialisation OR RNN
		if((initialisation_method_to_use == 1) || (initialisation_method_to_use == 2))
		{
			for(int j=0;j<population_size_global;j++)
			{
				//cout<<endl<<"Gene number"<<j<<endl;
				int curr_node = rand()%n;
				for(int i=0;i<n;i++){
					//cout<<curr_node<<"->";
					cities_order.push_back(curr_node);
					if(initialisation_method_to_use == 1)
						curr_node = findNearestNeighbour(curr_node, cities_order, this->cities_coords);
					else if(initialisation_method_to_use == 2)
					{
						double rndNumber = (double)rand() / (double)RAND_MAX;
						if(rndNumber < randomisation_in_RNN)
							curr_node = findRandomNeighbour(curr_node, cities_order, n);
						else
							curr_node = findNearestNeighbour(curr_node, cities_order, this->cities_coords);
					}
				}
				(this->population).push_back(cities_order);
				cities_order.clear();
			}
		}

		cout<<"Initialsed population"<<endl;
		//for(int i=0;i<n;i++)
		//	printChromosome((this->population)[i]);
		return;
	}
	
	for(int i=0;i<n;i++)
		cities_order.push_back(i);

	for(int i=0;i<population_size_global;i++)
	{
		for(int i=0;i<n/2;i++)
		{
			int r1 = rand()%n;
			int r2 = rand()%n;
			swap(cities_order[r1], cities_order[r2]);
		}	

		(this->population).push_back(cities_order);
	}

	cout<<"Initialsed population"<<endl;
	//for(int i=0;i<n;i++)
	//	printChromosome((this->population)[i]);
}

int Graph::calculateTotalDistanceOfAChromosome(vector<int> duplicate_chromosome){
	double sum_distances = 0;
	int n = this->num_vertices;
	iPair curr_pt, prev_pt;
	for(int j=0;j<n-1;j++){
		prev_pt = (this->cities_coords)[ duplicate_chromosome[j] ];
		curr_pt = (this->cities_coords)[ duplicate_chromosome[j+1] ];
		sum_distances += dist_2d(prev_pt, curr_pt);
		//cout<<"sum_distances: "<<sum_distances<<endl;
	}
	sum_distances = sum_distances + dist_2d((this->cities_coords)[ duplicate_chromosome[0] ], curr_pt);
	return sum_distances;
}

void Graph::CalculateFitness()
{
	//cout<<"Calculating Fitness"<<endl;
	int n = this->num_vertices;
	for(int i=0;i<population_size_global;i++)
	{
		//cout<<"i: "<<i<<endl;
		double sum_distances = 0;
		iPair curr_pt, prev_pt;
		for(int j=0;j<n-1;j++){
			//cout<<"j: "<<j<<endl;
			prev_pt = (this->cities_coords)[(this->population)[i][j] ];
			//cout<<"hi"<<endl;
			curr_pt = (this->cities_coords)[(this->population)[i][j+1] ];
			sum_distances += dist_2d(prev_pt, curr_pt);
			//cout<<"sum_distances: "<<sum_distances<<endl;
		}
		sum_distances = sum_distances + dist_2d( (this->cities_coords)[ population[i][0] ], curr_pt );
		total_distance[i] = sum_distances;
		fitness[i] = fitness_scaling_global* (double)(pow(sum_distances,-1));
	}
}

void Graph::DisplayBestSolution(int index)
{
	//cout<<"The best solution for iteration "<<index<<" is: ";
	int fittest_index;
	double max_fitness=-1111;
	for(int i=0;i<population_size_global;i++)
	{
		if( (this->fitness)[i] > max_fitness ){
			max_fitness = (this->fitness)[i];
			fittest_index = i;
		} 
	}

	this->fittest_index = fittest_index;
	this->best_fitness_generation = max_fitness;
	this->best_total_distance_generation = (this->total_distance)[fittest_index];

	//printChromosome((this->population)[fittest_index]);
	cout<<"Fitness value of the Fittest Chromosome of the generation: "<<(this->fitness)[fittest_index]<<endl;
	cout<<"Total distance: "<<(this->total_distance)[fittest_index]<<endl;

	if(max_fitness > this->best_fitness){
		this->best_fitness = max_fitness;
		this->best_total_distance = (this->total_distance)[fittest_index];
		best_chromosome = population[fittest_index];
	}
	//printChromosome(best_chromosome);
	cout<<"Fitness value of the Global Fittest Chromosome: "<<(this->best_fitness)<<endl;
	cout<<"Total distance: "<<(this->best_total_distance)<<endl;

	best_fitness_generation_list.push_back((this->fitness)[fittest_index]);
	best_chromosome_generation_list.push_back((this->population)[fittest_index]);
	best_fitness_global_list.push_back(this->best_fitness);
	best_chromosome_global_list.push_back(best_chromosome);
	best_total_distance_generation_list.push_back((this->total_distance)[fittest_index]);
	best_total_distance_global_list.push_back((this->best_total_distance));

}

//Roulette wheel selection is used here
void Graph::MakeNextGen()
{
	//cout<<"Making next generation of population"<<endl;
	vector<vector<int> > population_next_gen;
	vector<double> prob_list;
	double total_fitness=0;
	for(int i=0;i<population_size_global;i++)
		total_fitness += (this->fitness)[i];
	for(int i=0;i<population_size_global;i++)
		prob_list.push_back(( (this->fitness)[i] )/total_fitness);
	
	int for_loop_size = population_size_global;
	if(is_Elitism){
		for(int i=0;i<number_elites;i++)
			population_next_gen.push_back((this->best_chromosome_generation_list)[this->generation_number]);
		for_loop_size = for_loop_size - number_elites;
	}

	for(int j=0;j<for_loop_size;j++)
	{
		double rndNumber = (double)rand() / (double)RAND_MAX;
		double offset = 0.0;
		int pick = 0;

		for (int i = 0; i < population_size_global; i++) {
		    offset += prob_list[i];
		    if(rndNumber < offset){
		        pick = i;
		        break;
		    }
		}
		population_next_gen.push_back((this->population)[pick]);
	}
	
	prob_list.clear();
	(this->population) = population_next_gen;
	return;
}

//Elitism is not being followed
void Graph::CrossOver()
{
	if(!is_crossover_enabled)
		return;
	//cout<<"Doing CrossOver"<<endl;
	int n = (this->num_vertices);
	for(int i=0;i<population_size_global/2;i++)
	{
		double rand_num = (double)rand() / (double)RAND_MAX;
		if(rand_num > CrossOverProbability)
			continue;

		vector<int> reserved_indices;
		vector<int> entries_from_par2_to_par1;//during crossover
		vector<int> entries_from_par1_to_par2;
		int entr1_count=0;
		int entr2_count=0;
		int parent1 = rand()%population_size_global;
		int parent2 = rand()%population_size_global;
		if(( (parent1 == this->fittest_index) || (parent2 == this->fittest_index) ) && is_Elitism){
			//cout<<"Avoided crossover because of fittest_index: "<<this->fittest_index;
			continue;
		}
		//ensure co_site1 < co_site2
		int co_site1 = rand()%n;
		int co_site2 = rand()%n;
		if(co_site2 < co_site1)
			swap(co_site1, co_site2);
		else if(co_site1 == co_site2){
			if(co_site1 == n-1)
				co_site1 = n-2;
			else
				co_site2 = co_site1+1;
		}

		for(int j=co_site1;j<co_site2;j++)
			reserved_indices.push_back((this->population)[parent1][j]);

		for(int j=co_site1;j<co_site2;j++)
		{
			if( Iselem( ((this->population)[parent2][j]), reserved_indices ) )
				continue;
			else
				reserved_indices.push_back((this->population)[parent2][j]);
		}

		for(int j=0;j<n;j++)
		{
			if( Iselem( ((this->population)[parent1][j]), reserved_indices ) ) 
				entries_from_par1_to_par2.push_back( ((this->population)[parent1][j]) );
			if( Iselem( ((this->population)[parent2][j]), reserved_indices ) ) 
				entries_from_par2_to_par1.push_back( ((this->population)[parent2][j]) );
		}

		for(int j=0;j<n;j++)
		{
			if( Iselem( ((this->population)[parent1][j]), reserved_indices ) ){
				(this->population)[parent1][j] = entries_from_par2_to_par1[entr2_count];
				entr2_count++;
			}
			if( Iselem( ((this->population)[parent2][j]), reserved_indices ) ) {
				(this->population)[parent2][j] = entries_from_par1_to_par2[entr1_count];
				entr1_count++;
			}
		}

		reserved_indices.clear();
		entries_from_par2_to_par1.clear();
		entries_from_par1_to_par2.clear();
	}
}

double findMutationProbability_simulated_annealing(int l0, int l1){
	if(T_current < 0.01)
		return 0;
	double prob = exp((l0-l1)/T_current);
	return prob;
}

void Graph::Mutation()
{
	//cout<<"Doing Mutataion"<<endl;
	int n = (this->num_vertices);

	//Simulated annealing
	if(is_simulated_annealing){
		for(int i=0;i<population_size_global;i++)
		{
			for(int k=0;k<num_times_simulated_annealing_iteration;k++)
			{
				int site1 = rand()%n;
				int site2 = rand()%n;
				if(site2<site1)
					swap(site1, site2);
				else if(site1==site2){
					if(site1==n-1)
						site1--;
					else
						site2++;
				}

				//we have site1, site2...Now we wanna switch the genes between these two sites
				vector<int> swapped_genes;
				for(int j=site2-1;j>site1;j--)
					swapped_genes.push_back((this->population)[i][j]);
				int count =0;
				vector<int> duplicate_chromosome = (this->population)[i];
				for(int j=site1+1;j<site2;j++){
					duplicate_chromosome[j] = swapped_genes[count];
					count++;
				}

				if(calculateTotalDistanceOfAChromosome(duplicate_chromosome) < calculateTotalDistanceOfAChromosome( (this->population)[i]))
					(this->population)[i] = duplicate_chromosome;	

				else if( calculateTotalDistanceOfAChromosome(duplicate_chromosome) > calculateTotalDistanceOfAChromosome( (this->population)[i]) ){
					double l0 = calculateTotalDistanceOfAChromosome((this->population)[i]);
					double l1 = calculateTotalDistanceOfAChromosome(duplicate_chromosome);
					double prob_mut = findMutationProbability_simulated_annealing(l0,l1);
					duplicate_chromosome.clear();
					swapped_genes.clear();
				}
			}	
		}	

		T_current = T_current - T_rate_of_decrease;	
		return;
	}

	//If RBINBNNM(custom mutation for TSP) is enabled
	if(is_RBIBNNM_mutation){
		for(int i=0;i<population_size_global;i++){
			for(int j=0;j<n;j++)
			{
				double rndNumber = (double)rand() / (double)RAND_MAX;
				if(rndNumber < MutationProbability){
					int rand_k = rand()%n;
					vector<int> cities_order;
					cities_order.push_back(rand_k);
					int index_NN = findNearestNeighbour(rand_k, cities_order, this->cities_coords);
					int index_to_be_swapped;
					if(index_NN>0)
						index_to_be_swapped = index_NN-1;
					else
						index_to_be_swapped = index_NN+1;
					swap( (this->population)[i][index_to_be_swapped] , (this->population)[i][rand_k] );
					cities_order.clear();
				}
			}	
		}	
		return;
	}

	if(is_custom_mutation){
		for(int i=0;i<population_size_global;i++)
		{
			double rndNumber = (double)rand() / (double)RAND_MAX;
			if(rndNumber > MutationProbability)
				continue;

			for(int k=0;k<num_iterns_custom_mutation;k++)
			{
				int site1 = rand()%n;
				int site2 = rand()%n;
				if(site2<site1)
					swap(site1, site2);
				else if(site1==site2){
					if(site1==n-1)
						site1--;
					else
						site2++;
				}
				//we have site1, site2...Now we wanna switch the genes between these two sites
				vector<int> swapped_genes;
				for(int j=site2-1;j>site1;j--)
					swapped_genes.push_back((this->population)[i][j]);
				int count =0;
				vector<int> duplicate_chromosome = (this->population)[i];
				for(int j=site1+1;j<site2;j++){
					duplicate_chromosome[j] = swapped_genes[count];
					count++;
				}

				if( calculateTotalDistanceOfAChromosome(duplicate_chromosome) > calculateTotalDistanceOfAChromosome((this->population)[i]) ){
					duplicate_chromosome.clear();
					swapped_genes.clear();
					continue;
				}
				//count=0;
				(this->population)[i] = duplicate_chromosome;	
			}		

		}	
		return;
	}

	for(int i=0;i<population_size_global;i++){
		for(int j=0;j<n;j++){
			double rndNumber = (double)rand() / (double)RAND_MAX;
			if(rndNumber < MutationProbability){
				int rand_k = rand()%n;
				swap( (this->population)[i][j] , (this->population)[i][rand_k] );
			}
		}
		
	}
}	

ofstream local_chromosome_list;
ofstream global_chromosome_list;
ofstream local_fitness_list;
ofstream global_fitness_list;
ofstream local_total_distance_list;
ofstream global_total_distance_list;

void Graph::TSP()
{
	int n = this->num_vertices;
	cout<<"Initialsing population, Plz wait!!..............."<<endl;
	InitialisePopulation();
	cout<<"Initialsed population"<<endl;
	int i;
	for(i=0;i<num_iterations_global;i++)
	{
		this->generation_number = i;
		cout<<endl<<endl<<"Iteration number: "<<i<<endl;
		//int rand_k = rand()%n;
		//cout<<"rand_k: "<<rand_k<<endl;
		CalculateFitness();
		DisplayBestSolution(i);
		MakeNextGen();//changed population to next generation by selection by roulet wheel
		CrossOver();
		Mutation();

		for(int j=0;j<(this->best_chromosome_generation_list)[i].size();j++)
    		local_chromosome_list<<(this->best_chromosome_generation_list)[i][j]<<" ";
    	local_chromosome_list<<endl;

    	for(int j=0;j<(this->best_chromosome_global_list)[i].size();j++)
    		global_chromosome_list<<(this->best_chromosome_global_list)[i][j]<<" ";
    	global_chromosome_list<<endl;

    	local_fitness_list<<(this->best_fitness_generation_list)[i]<<endl;
    	global_fitness_list<<(this->best_fitness_global_list)[i]<<endl;

    	local_total_distance_list<<(this->best_total_distance_generation_list)[i]<<endl;
    	global_total_distance_list<<(this->best_total_distance_global_list)[i]<<endl;

    	clock_t end_time = clock();
		double cpu_time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;
		printf("Time: %f \n",cpu_time_used);
	}

	cout<<endl<<endl<<"THE FINAL SOLUTION IS: "<<endl;
	DisplayBestSolution(i);

	cout<<endl<<endl<<"THE GLOBAL BEST SOLUTION TILL NOW IS: "<<endl;
	printChromosome(this->best_chromosome);

	local_chromosome_list.close();
	global_chromosome_list.close();
	local_fitness_list.close();
	global_fitness_list.close();
	local_total_distance_list.close();
	global_total_distance_list.close();
}

int main()
{
	srand(time(0)); 
	clock_t end_time;
	double cpu_time_used;
	start_time = clock();

	ifstream fin; 
	fin.open("city_coord_data.txt");
	iPair temp_point;
	vector<iPair> city;
	fin>>temp_point.first;
	while(!fin.eof())
	{
	  fin>>temp_point.second;
	  city.push_back(temp_point);
	  fin>>temp_point.first;
	}
	cout<<"Number of cities are:"<<city.size()<<endl;

	Graph g(city);
	//g.InitialisePopulation();
	local_chromosome_list.open("best_chromosome_generation.txt");
	global_chromosome_list.open("best_chromosome_global.txt");
	local_fitness_list.open("best_fitness_generation.txt");
	global_fitness_list.open("best_fitness_global.txt");
	local_total_distance_list.open("best_total_distance_generation.txt");
	global_total_distance_list.open("best_total_distance_global.txt");

	g.TSP();

	end_time = clock();
    cpu_time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;
  
    printf("TSP implementation took %f seconds to execute \n", cpu_time_used); 
    return 0; 
}


