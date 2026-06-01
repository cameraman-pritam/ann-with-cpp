// The Required libraries
#include <algorithm>
#include <fstream>
#include <ios>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>
using namespace std;

// Useful Keywords
using vctr = vector<float>;   // defines a vector containing floats
using mtrx = vector<vctr>;    // defines a 2D vector [vector containing vectors, where each vector
                              // contains a float]
constexpr float zero = 0.0F;  // handy iteration of true zero in float

/* Creates a single vector of the actual required result (5 is turned into a vector where 6th term
 * (5) is 1.0 `one_hot[5] = 1.0`)*/
vctr createOneHotVector(int label, int classes = 10)
{
  vctr one_hot(classes, zero);
  if (label >= 0 && label < classes)
  {
    one_hot[label] = 1.0F;
  }
  return one_hot;
}

// Loads ALL rows — returns false if file couldn't be opened
bool loadCSV(const string& filename, vector<vctr>& inputs, vector<vctr>& actuals)
{
  ifstream file(filename);
  if (!file.is_open())
  {
    cerr << "[ERROR] Could not open: " << filename << "\n";
    return false;
  }

  string line;
  int count = 0;

  while (getline(file, line))  // basically for each line
  {
    if (line.empty())
    {
      continue;
    }

    stringstream ss(line);  // create a stringstream of the line to work on as a file
    string value;

    // First column is the label
    getline(ss, value, ',');  // Take input as long as u dont get a comma (,)
    int label = stoi(value);  // Turn the string into a integer

    // Remaining 784 columns are pixels
    vctr pixels;
    pixels.reserve(784);             // Already keep 784 spaces ready
    while (getline(ss, value, ','))  // basically a for loop
    {
      pixels.push_back(
          stof(value)
          / 255.0F);  // Store the value in pixels (Normalized so that values are b/w 0.0 and 1.0)
    }

    inputs.push_back(std::move(pixels));  // move the entire thing directly instead of copying
    actuals.push_back(createOneHotVector(label));  // store the label vector

    ++count;
    if (count % 10000 == 0)
      cout << "  Loaded " << count << " samples...\n";
  }

  cout << "Total samples loaded: " << count << "\n";
  return true;
}

// ─── ANN Class ──────────────────────────────────────────────────────────────

class ANN
{
  private:              // Only the ai can touch its brain
  mtrx mat;             // The 2D vector matrix
  vctr bias;            // the bias vector
  vctr last_z;          // The last pre-activation values
  float learning_rate;  // the rate at which the ai learns

  // Create a garbage random matrix to be worked on
  mtrx initMatrix(int rows, int cols)
  {
    mtrx m(rows, vctr(cols, zero));                     // create a matrix
    mt19937 gen(42);                                    // random numbers generation
    uniform_real_distribution<float> dis(-0.1F, 0.1F);  // distribution chart b/w -0.1 and 0.1
    for (auto& row : m)
      for (auto& val : row)
        val = dis(gen);
    return m;
  }

  vctr initVector(int size)  // same as initmatrix but for the bias vector
  {
    vctr v(size, zero);
    mt19937 gen(42);
    uniform_real_distribution<float> dis(-0.1F, 0.1F);
    for (auto& val : v)
      val = dis(gen);
    return v;
  }

  vctr mtrxMul(const mtrx& m, const vctr& v)  // multiply the Weights with the Input
  {
    int rows = static_cast<int>(m.size());
    int cols = static_cast<int>(m[0].size());
    vctr result(rows, zero);
    for (int i = 0; i < rows; i++)
      for (int j = 0; j < cols; j++)
        result[i] += m[i][j] * v[j];  // multiply each weight with corresponding input
    return result;
  }

  vctr reluDeriv(const vctr& z)
  {
    vctr d(z.size(), zero);
    for (size_t i = 0; i < z.size(); i++)
      d[i] = (z[i] > 0.0F) ? 1.0F : 0.01F;  // derivation of the relu neurons;
    return d;
  }

  public:
  // ─── Inside ANN class: add these two public methods ───────────────────────────

  // Returns which slot has the highest value — that's the predicted digit
  int argmax(const vctr& v)
  {
    return static_cast<int>(max_element(v.begin(), v.end()) - v.begin());
  }

  // Runs the full test set, prints accuracy
  void evaluate(const vector<vctr>& inputs, const vector<vctr>& actuals)
  {
    int correct = 0;
    int total = static_cast<int>(inputs.size());

    for (int i = 0; i < total; i++)
    {
      vctr output = predict(inputs[i]);
      int predicted = argmax(output);
      int expected = argmax(actuals[i]);  // actual is one-hot, argmax gives label back

      if (predicted == expected)
        ++correct;
    }

    float accuracy = 100.0F * static_cast<float>(correct) / static_cast<float>(total);
    cout << "Accuracy: " << correct << "/" << total << "  (" << accuracy << "%)\n";
  }

  ANN(int output_size, int input_size, float lr = 0.5F)
      : learning_rate(lr)
  {
    mat = initMatrix(output_size, input_size);
    bias = initVector(output_size);
  }

  bool loadTraining(const string& filename)
  {
    ifstream file(filename, ios::binary);
    if (!file.is_open())
    {
      cerr << "[WARNING] No saved model found. Starting fresh.\n";
      return false;
    }

    int rows, cols;
    file.read(reinterpret_cast<char*>(&rows), sizeof(int));
    file.read(reinterpret_cast<char*>(&cols), sizeof(int));

    mat.assign(rows, vctr(cols, 0.0F));
    for (int i = 0; i < rows; i++)
      file.read(reinterpret_cast<char*>(mat[i].data()), cols * sizeof(float));

    int bias_size;
    file.read(reinterpret_cast<char*>(&bias_size), sizeof(int));
    bias.resize(bias_size);
    file.read(reinterpret_cast<char*>(bias.data()), bias_size * sizeof(float));

    cout << "Model loaded from: " << filename << "\n";
    return true;
  }

  void saveTraining(const string& filename)
  {
    ofstream file(filename, ios::binary);

    int rows = static_cast<int>(mat.size());
    int cols = static_cast<int>(mat[0].size());
    file.write(reinterpret_cast<const char*>(&rows), sizeof(int));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(int));

    for (int i = 0; i < rows; i++)
      file.write(reinterpret_cast<const char*>(mat[i].data()), cols * sizeof(float));

    int bias_size = static_cast<int>(bias.size());
    file.write(reinterpret_cast<const char*>(&bias_size), sizeof(int));
    file.write(reinterpret_cast<const char*>(bias.data()), bias_size * sizeof(float));

    cout << "Model saved to: " << filename << "\n";
  }

  vctr predict(const vctr& input)  // applies LeakyReLU
  {
    last_z = mtrxMul(mat, input);
    for (size_t i = 0; i < last_z.size(); i++)
      last_z[i] += bias[i];

    vctr activated = last_z;
    for (auto& val : activated)
      val = (val > 0.0F) ? val : 0.01F * val;

    return activated;
  }

  float loss(const vctr& predicted, const vctr& actual)
  {
    float total = zero;
    for (size_t i = 0; i < predicted.size(); i++)
    {
      float err = predicted[i] - actual[i];
      total += err * err;
    }
    return total / static_cast<float>(predicted.size());
  }

  void backPropagate(const vctr& input, const vctr& predicted, const vctr& actual)
  {
    vctr relu_d = reluDeriv(last_z);

    for (size_t i = 0; i < mat.size(); i++)
    {
      float delta = (predicted[i] - actual[i]) * relu_d[i];

      for (size_t j = 0; j < mat[0].size(); j++)
        mat[i][j] -= learning_rate * delta * input[j];

      bias[i] -= learning_rate * delta;
    }
  }
};

// ─── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
  const string model_file = "model.bin";
  ANN ann(10, 784, 0.02F);

  // Default mode is test if no arg given — you've already trained
  string mode = (argc > 1) ? argv[1] : "test";

  if (mode == "train")
  {
    cout << "Loading MNIST training data...\n";
    vector<vctr> inputs, actuals;
    if (!loadCSV("mnist_train.csv", inputs, actuals))
    {
      cerr << "[FATAL] Could not load training data.\n";
      return 1;
    }

    bool loaded = ann.loadTraining(model_file);
    if (!loaded)
      cout << "Fresh model initialised.\n";

    size_t n = inputs.size();
    vector<size_t> indices(n);
    iota(indices.begin(), indices.end(), 0);
    mt19937 rng(42);

    int epochs = 10;
    cout << "\n---Training (" << n << " samples, " << epochs << " epochs)---\n";

    for (int epoch = 1; epoch <= epochs; epoch++)
    {
      shuffle(indices.begin(), indices.end(), rng);
      float total_loss = 0.0F;

      for (size_t idx : indices)
      {
        vctr predicted = ann.predict(inputs[idx]);
        total_loss += ann.loss(predicted, actuals[idx]);
        ann.backPropagate(inputs[idx], predicted, actuals[idx]);
      }

      cout << "Epoch " << epoch << "/" << epochs
           << " | Avg Loss: " << (total_loss / static_cast<float>(n)) << "\n";

      ann.saveTraining(model_file);
    }

    cout << "\nTraining complete.\n";
  }
  else if (mode == "test")
  {
    cout << "Loading MNIST test data...\n";
    vector<vctr> inputs, actuals;
    if (!loadCSV("mnist_test.csv", inputs, actuals))
    {
      cerr << "[FATAL] Could not load test data.\n";
      return 1;
    }

    if (!ann.loadTraining(model_file))
    {
      cerr << "[FATAL] No saved model found. Train first.\n";
      return 1;
    }

    cout << "\n---Evaluating on test set---\n";
    ann.evaluate(inputs, actuals);
  }
  else if (mode == "predict")
  {
    // Pass a CSV row directly: ./ann predict "5,128,0,255,..."
    if (argc < 3)
    {
      cerr << "Usage: ./ann predict \"label,px1,px2,...\"\n";
      return 1;
    }

    if (!ann.loadTraining(model_file))
    {
      cerr << "[FATAL] No saved model found. Train first.\n";
      return 1;
    }

    // Parse the raw CSV string passed as argument
    vctr pixels;
    int label = -1;
    stringstream ss(argv[2]);
    string value;

    getline(ss, value, ',');
    label = stoi(value);

    while (getline(ss, value, ','))
      pixels.push_back(stof(value) / 255.0F);

    vctr output = ann.predict(pixels);
    int predicted = ann.argmax(output);

    cout << "Expected : " << label << "\n";
    cout << "Predicted: " << predicted << "\n\n";
    cout << "Raw output scores:\n";
    for (int i = 0; i < 10; i++)
      cout << "  Digit " << i << ": " << output[i] << "\n";
  }
  else
  {
    cerr << "Unknown mode. Use: train | test | predict\n";
    return 1;
  }

  return 0;
}