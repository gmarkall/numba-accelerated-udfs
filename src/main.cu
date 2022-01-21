/* Copyright (c) 2022, NVIDIA CORPORATION.

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* A very simple demo application using the Filigree library - it reads in an
 * image, converts it to greyscale and writes it back out again.
 */

#include "argparse/argparse.hpp"
#include "filigree.hxx"
#include "rmm/mr/device/pool_memory_resource.hpp"
#include <iostream>

argparse::ArgumentParser parse_args(int argc, char *argv[])
{
  argparse::ArgumentParser program("greyscaler");

  program.add_argument("input").help("Input image file name");
  program.add_argument("output").help("Output image file name");

  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    std::exit(1);
  }

  return program;
}

int main(int argc, char *argv[]) {
  // Parse arguments
  argparse::ArgumentParser args = parse_args(argc, argv);

  // Initialize RMM
  rmm::mr::cuda_memory_resource cmr;
  rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> mr(&cmr);
  rmm::cuda_stream stream;

  std::string input_filename = args.get<std::string>("input");
  std::string output_filename = args.get<std::string>("output");

  std::cout << "Processing " << input_filename << "..." << std::endl;

  filigree::Image *img;

  try {
    img = new filigree::Image{input_filename, &mr, stream};
  } catch (const std::exception &e) {
    std::cerr << "Error loading " << input_filename << ": ";
    std::cerr << e.what() << std::endl;
    return 1;
  }

  std::cout << "Converting to greyscale..." << std::endl;

  img->to_greyscale();

  std::cout << "Writing to " << output_filename << "..." << std::endl;

  img->write(output_filename);

  std::cout << "Finished writing" << std::endl;

  return 0;
}
