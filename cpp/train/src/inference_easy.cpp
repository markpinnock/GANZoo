#include "loaders.h"
#include "json.h"

#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/tag_constants.h"


int main(int argc, char** argv)
{
	const std::string model_path{ "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/008_GAN_Face_Prac/models/DCGANTest1/generator" };
	tf::SavedModelBundle model;
	TF_CHECK_OK(tf::LoadSavedModel(tf::SessionOptions(), tf::RunOptions(), model_path, { tf::kSavedModelTagServe}, &model));

	std::vector<Tensor> generator_output;
	std::vector<Tensor> noise;

	tf::Scope input_scope{ tf::Scope::NewRootScope() };
	Output noise_op = ops::RandomNormal(input_scope.WithOpName("random_noise"), tf::Input({ 1, 128 }), tf::DT_FLOAT);
	TF_CHECK_OK(input_scope.status());

	tf::ClientSession session(input_scope);
	TF_CHECK_OK(session.Run({ noise_op }, &noise));

	TF_CHECK_OK(model.GetSession()->Run({ {"serving_default_input_1:0", noise[0]} }, { "StatefulPartitionedCall:0" }, {}, &generator_output));
	TF_CHECK_OK(utils::WriteImage("./output/test.png", generator_output));
	generator_output.clear();

	return EXIT_SUCCESS;
}