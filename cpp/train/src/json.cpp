#include "json.h"

#include <algorithm>
#include <fstream>
#include <iostream>

//------------------------------------------------------------------------

void json::Read(std::string file_path) // std::string& s, json::Params* p
{
	std::string major_delim{ '{' };
	std::string minor_delim{ ':' };
	std::vector<std::string> params;

	std::ifstream fin(file_path.c_str());
	std::string data;
	std::string substring;

	if (fin.fail())
	{
		std::cerr << "File not found: " << file_path << std::endl;
	}

	fin.seekg(0, std::ios::end);
	data.resize(fin.tellg());
	fin.seekg(0, std::ios::beg);
	fin.read(&data[0], data.size());
	fin.close();

	size_t prev{ 0 };
	size_t curr{ 0 };

	//while (curr != std::string::npos)
	//{
	//	curr = data.find(major_delim, prev);
	//	substring = data.substr(prev, curr - prev);
	//	//substring.erase(
	//	//	std::remove(substring.begin(), substring.end(), '{'),
	//	//	substring.end());
	//	//substring.erase(
	//	//	std::remove(substring.begin(), substring.end(), '\n'),
	//	//	substring.end());
	//	//substring.erase(
	//	//	std::remove(substring.begin(), substring.end(), '\t'),
	//	//	substring.end());
	//	//substring.erase(
	//	//	std::remove(substring.begin(), substring.end(), '}'),
	//	//	substring.end());
	//	//substring.erase(
	//	//	std::remove(substring.begin(), substring.end(), ' '),
	//	//	substring.end());
	//	params.push_back(substring);
	//	prev = curr + 1;
	//}
	data.erase(
		std::remove(data.begin(), data.end(), '\n'),
		data.end());
	data.erase(
		std::remove(data.begin(), data.end(), '\t'),
		data.end());
	data.erase(
		std::remove(data.begin(), data.end(), ' '),
		data.end());

	std::cout << data << std::endl;
}