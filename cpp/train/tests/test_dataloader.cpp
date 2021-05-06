#include "../src/dataloader.h"

#include "gtest/gtest.h"


//class DataloaderTest : public ::testing::Test
//{
//protected:
//	void SetUp() override
//	{
//		
//	}
//
//	Dataloader dataloader;
//	tf::Status status;
//};

TEST(DataloaderInit, LoadFileNames)
{
	// Get list of images in test folder
	std::string file_path{ "./data/" };
	tf::Env* env = tf::Env::Default();
	std::vector<std::string> file_names;
	ASSERT_TRUE(env->GetChildren(file_path, &file_names).ok());

	for (auto& el : file_names)
	{
		el = file_path + el;
	}

	// Load filenames using Dataloader and test
	Dataloader d(4);
	tf::Status s = d.loadFilenames(file_path);
	ASSERT_TRUE(s.ok());
	ASSERT_EQ(file_names.size(), d.getFilenames().size());

	for (int i{ 0 }; i < file_names.size(); ++i)
	{
		EXPECT_EQ(file_names[i], d.getFilenames()[i]);
	}
}


int main(int argc, char** argv)
{
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}