#include "dataloader.h"
#include "loaders.h"
#include "json.h"


int main(int argc, char** argv)
{
	Dataloader d(8);
	if (!d.loadFilenames("./").ok()) return EXIT_FAILURE;
	std::vector<Tensor> t;
	d.getMinibatch(t);
	std::cout << t.size() << t[0].shape().DebugString() << std::endl;

	return EXIT_SUCCESS;
}