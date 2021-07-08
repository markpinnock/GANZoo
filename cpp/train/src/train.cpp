#include "dataloader.h"
#include "DCGAN.h"
#include "loaders.h"
#include "json.h"


int main(int argc, char** argv)
{
	Dataloader d(64);
	//TF_CHECK_OK(d.loadFilenames("../../../../tests/data/"));
	TF_CHECK_OK(d.loadFilenames("D:/CelebAMask-HQ/CelebA-HQ-img/"));
	TF_CHECK_OK(d.createImageGraph(32, 32));

	DCGAN model(32, 100);

	TF_CHECK_OK(model.discriminatorTrainingGraph(32));
	TF_CHECK_OK(model.generatorTrainingGraph(32));
	TF_CHECK_OK(model.createDiscOptimiser(0.001, 0.5, 0.999));
	TF_CHECK_OK(model.createGenOptimiser(0.001, 0.5, 0.999));
	TF_CHECK_OK(model.initialiseModels());
	//TF_CHECK_OK(model.getLatentNoise(noise, 4));
	model.printModel();

	//TF_CHECK_OK(model.runGenerator(noise, g_output));
	
	std::vector<Tensor> real_minibatch;
	std::vector<float> discriminator_losses;
	std::vector<float> generator_losses;
	std::vector<Tensor> generator_output;

	for (int i{ 0 }; i < 10; ++i)
	{
		for (int j{ 0 }; j < d.getNumMinibatches(); ++j)
		{
			TF_CHECK_OK(d.getMinibatch(real_minibatch));
			TF_CHECK_OK(model.trainStep(real_minibatch, discriminator_losses, generator_losses));
			std::cout << i << " " << j << " "
				<< discriminator_losses[discriminator_losses.size() - 1]
				<< " "
				<< generator_losses[generator_losses.size() - 1]
				<< std::endl;
		}

		TF_CHECK_OK(model.runGenerator(generator_output));
		TF_CHECK_OK(utils::WriteImage("./output/" + std::to_string(i) + ".png", generator_output));
		generator_output.clear();
	}

	return EXIT_SUCCESS;
}