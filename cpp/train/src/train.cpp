#include "dataloader.h"
#include "loaders.h"
#include "json.h"


int main(int argc, char** argv)
{
	Dataloader d(8);
	TF_CHECK_OK(d.loadFilenames("../../../../tests/data/"));
	TF_CHECK_OK(d.createImageGraph(64, 64));
	std::vector<std::string> a = d.getFilenames();

	for (auto el : a)
	{
		std::cout << el << std::endl;
	}
	
	std::vector<Tensor> t;
	d.getMinibatch(t);
	std::cout << t.size() << " " << t[0].shape().DebugString() << std::endl;

	return EXIT_SUCCESS;
}