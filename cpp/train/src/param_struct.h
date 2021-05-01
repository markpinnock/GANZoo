#ifndef PARAM_STRUCT_H
#define PARAM_STRUCT_H

#include <string>
#include <vector>

namespace json
{
	struct Params
	{
		// Expt
		std::string save_path;
		std::string data_path;
		std::string expt_name;
		int num_examples{ 16 };
		bool verbose{ true };
		int dataset_size{ 0 };
		bool from_RAM{ true }; // Needed?
		int save_every{ 0 };
		int scales{ 64 };
		int epochs{ 500 };
		int mb_size{ 16 };

		// Hyperparameters
		std::string model;
		std::string loss_fn;
		int n_critic{ 1 };
		std::string g_opt{ "Adam" };
		std::string d_opt{ "Adam" };
		std::vector<float> g_eta{ 2e-4, 0.5, 0.999 };
		std::vector<float> d_eta{ 2e-4, 0.5, 0.999 };
		int ndf{ 16 };
		int ngf{ 16 };
		bool d_dense{ false };
		bool g_dense{ false };
		std::string d_act;
		std::string g_act;
		int latent_dim{ 128 };
		int max_channels{ 128 };
		int max_res{ 64 };
		bool augment{ false };
		float ema_beta{ 0.99f };
		std::string g_out;
		bool add_noise{ false };
	};
}

#endif // PARAM_STRUCT_H

