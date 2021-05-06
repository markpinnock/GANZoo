#include "../src/loaders.h"

#include "gtest/gtest.h"


class ImgLoadTest : public ::testing::Test
{
protected:
	void SetUp() override
	{
		tf::string file_path{ "./data/Lena.png" };
		tf::Status status = utils::ReadImage(file_path, 256, 256, &test_img);
	}

	std::vector<Tensor> test_img;
	tf::Status status;
};

TEST_F(ImgLoadTest, StatusTest)
{
	ASSERT_EQ(status.ok(), true);
}

TEST_F(ImgLoadTest, DimTest)
{
	std::vector<int> gt_dims{ 1, 256, 256, 3 };

	EXPECT_EQ(test_img[0].dims(), gt_dims.size());

	for (int i{ 0 }; i < test_img[0].dims(); ++i)
	{
		EXPECT_EQ(test_img[0].shape().dim_size(i), gt_dims[i]);
	}
}

TEST_F(ImgLoadTest, PixelTest)
{
	auto root = tf::Scope::NewRootScope();
	auto img_ph = ops::Placeholder(root.WithOpName("img"), tf::DT_FLOAT);
	auto dims = ops::Const(root.WithOpName("dims"), { 1, 2, 3 });
	auto min = ops::Min(root.WithOpName("min"), img_ph, dims);
	auto max = ops::Max(root.WithOpName("max"), img_ph, dims);

	std::vector<Tensor> min_out;
	std::vector<Tensor> max_out;
	tf::ClientSession session(root);

	TF_CHECK_OK(session.Run({ {img_ph, test_img[0]} }, { min }, &min_out));
	TF_CHECK_OK(session.Run({ {img_ph, test_img[0]} }, { max }, &max_out));
	EXPECT_EQ(min_out[0].flat<float>()(0), -1) << min_out[0].flat<float>()(0);
	EXPECT_EQ(max_out[0].flat<float>()(0), 1) << max_out[0].flat<float>()(0);
}

TEST_F(ImgLoadTest, WriteTest)
{
	tf::Status status = utils::WriteImage("./data/out.png", test_img);
	EXPECT_EQ(status.ok(), true);
}

int main(int argc, char** argv)
{
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}