#include "dataloader.h"
#include "loaders.h"
#include "json.h"


int main(int argc, char** argv)
{
	Dataloader d(10);
	TF_CHECK_OK(d.loadFilenames("../../../../tests/data/"));
	TF_CHECK_OK(d.createImageGraph(64, 64));

	std::vector<Tensor> t;

	for (int i{ 0 }; i < d.getNumMinibatches(); ++i)
	{
		TF_CHECK_OK(d.getMinibatch(t));
		std::cout << t.size() << " " << t[0].shape().DebugString() << " " << t[0].shape().dim_size(0) << std::endl;
	}

	TF_CHECK_OK(d.getMinibatch(t));
	std::cout << t.size() << " " << t[0].shape().DebugString() << " " << t[0].shape().dim_size(0) << std::endl;

	return EXIT_SUCCESS;
}