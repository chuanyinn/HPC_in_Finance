#include <mkl_vsl.h>
#include <tbb/task_group.h>

#include <iostream>

class PiTask
{
public:
    long num_points;
    long& count_points_in_circle;

    VSLStreamStatePtr stream;

    PiTask(long numpoints, VSLStreamStatePtr stream, long& in)
        : num_points(numpoints), stream(stream)
		, count_points_in_circle(in)     
	{
        count_points_in_circle = 0; // make sure to initialize to zero.
    }

    void operator()() const //has to be const
    {
        const int block_size = 1000; //to represent 1000 random points
    
        double rands[2 * block_size]; //each point has 2 coordinates (x,y)
         
        int nblocks = num_points / block_size;
        
        //nblocks
        for (int j = 0; j < nblocks; j++) 
        {
            vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, 
				stream, 
				2 * block_size,  
				rands, 
				-1.0, 1.0); //range [-1,1]
            
            for (int i = 0; i < block_size; i++) 
            {
                double x = rands[2 * i + 0];
                double y = rands[2 * i + 1];
                if (x * x + y * y <= 1.0) ++(count_points_in_circle);
            }
        }
    }
};


int main() 
{
    int errorcode;
    //we devide 500 million simulations among 50 tasks
    const unsigned long num_tasks = 50;
    const unsigned long samples_per_task = 10000000l;
    int seed = 777;

    VSLStreamStatePtr stream[num_tasks];

    for (int i = 0; i < num_tasks; i++) 
    {
        errorcode = vslNewStream(&stream[i], VSL_BRNG_MCG59, seed);
        if (errorcode) return 1;

        errorcode = vslLeapfrogStream(stream[i], i, num_tasks);

        if (errorcode) return 1;
    }

    tbb::task_group group;
    long points_inside_circle_per_task[num_tasks];


    for (int i = 0; i < num_tasks; i++) 
    {
        group.run(PiTask(samples_per_task, 
			stream[i],
			points_inside_circle_per_task[i]));
    }

    group.wait();

    unsigned long total_num_points_inside_circle = 0ul;

    for (int i = 0; i < num_tasks; i++)
    {
		total_num_points_inside_circle += points_inside_circle_per_task[i];
    }

	unsigned long total_num_points  = samples_per_task * num_tasks;

    std::cout << "pi = " << 4.0 * total_num_points_inside_circle / total_num_points << std::endl;

    for (int i = 0; i < num_tasks; i++)
    {
        vslDeleteStream(&stream[i]);
    }
}
