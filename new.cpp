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
using vctr = vector<float>;
using mtrx = vector<vctr>;
constexpr float zero = 0.0F;

/* Creates a single vector of the actual required result (5 is turned into a vector where 6th term
 * (5) is 1.0 `one_hot[5] = 1.0`)*/
vctr createOneHotVector(int label, int classes = 10)
{
  vctr one_hot(classes, zero);
  if (label >= 0 && label < classes)
    one_hot[label] = 1.0F;
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

  while (getline(file, line))
  {
    if (line.empty())
      continue;

    stringstream ss(line);
    string value;

    getline(ss, value, ',');
    int label = stoi(value);

    vctr pixels;
    pixels.reserve(784);
    while (getline(ss, value, ','))
      pixels.push_back(stof(value) / 255.0F);

    inputs.push_back(std::move(pixels));
    actuals.push_back(createOneHotVector(label));

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
  private:
  // ── Layer storage ──────────────────────────────────────────────────────────
  // Instead of a single mat/bias, we now have one entry per layer.
  // Layer 0: 256×784   (input → first hidden)
  // Layer 1: 128×256   (first hidden → second hidden)
  // Layer 2:  64×128   (second hidden → third hidden)
  // Layer 3:  10×64    (third hidden → output)
  vector<mtrx> weights;  // weights[L] is the weight matrix for layer L
  vector<vctr> biases;   // biases[L] is the bias vector for layer L

  // During forward pass we need to remember two things per layer for backprop:
  //   z        = raw weighted sum BEFORE activation  (used for relu derivative)
  //   a        = activated output AFTER activation   (used as input to next layer)
  vector<vctr> last_z;  // last_z[L]  — pre-activation values at layer L
  vector<vctr> last_a;  // last_a[L]  — post-activation values at layer L
                        // last_a[0] is just the input itself (handy trick)

  float learning_rate;  // current learning rate — will shrink over time

  // ── Helpers ────────────────────────────────────────────────────────────────
  mtrx initMatrix(int rows, int cols)
  {
    mtrx m(rows, vctr(cols, zero));
    // He initialisation: scale by sqrt(2/fan_in) — works better with ReLU
    // than the old uniform(-0.1, 0.1) because it keeps variance stable as
    // the signal passes through many layers.
    mt19937 gen(42);
    float stddev = sqrt(2.0F / static_cast<float>(cols));
    normal_distribution<float> dis(0.0F, stddev);
    for (auto& row : m)
      for (auto& val : row)
        val = dis(gen);
    return m;
  }

  vctr initVector(int size)
  {
    // Biases start at zero — standard practice with He init
    return vctr(size, zero);
  }

  vctr mtrxMul(const mtrx& m, const vctr& v)
  {
    int rows = static_cast<int>(m.size());
    int cols = static_cast<int>(m[0].size());
    vctr result(rows, zero);
    for (int i = 0; i < rows; i++)
      for (int j = 0; j < cols; j++)
        result[i] += m[i][j] * v[j];
    return result;
  }

  // Leaky ReLU derivative: 1.0 if z>0, else 0.01
  vctr reluDeriv(const vctr& z)
  {
    vctr d(z.size(), zero);
    for (size_t i = 0; i < z.size(); i++)
      d[i] = (z[i] > 0.0F) ? 1.0F : 0.01F;
    return d;
  }

  // Leaky ReLU activation
  vctr relu(const vctr& z)
  {
    vctr a(z.size());
    for (size_t i = 0; i < z.size(); i++)
      a[i] = (z[i] > 0.0F) ? z[i] : 0.01F * z[i];
    return a;
  }

  public:
  // ── Constructor ────────────────────────────────────────────────────────────
  // topology: a list of layer sizes INCLUDING input and output.
  // e.g. {784, 256, 128, 64, 10} means:
  //   layer 0 weights: 256×784
  //   layer 1 weights: 128×256
  //   layer 2 weights:  64×128
  //   layer 3 weights:  10×64
  ANN(const vector<int>& topology, float lr = 0.01F)
      : learning_rate(lr)
  {
    int num_layers = static_cast<int>(topology.size()) - 1;  // number of weight layers

    weights.resize(num_layers);
    biases.resize(num_layers);
    last_z.resize(num_layers);
    last_a.resize(num_layers + 1);  // +1 to also store the raw input at last_a[0]

    for (int l = 0; l < num_layers; l++)
    {
      weights[l] = initMatrix(topology[l + 1], topology[l]);
      biases[l] = initVector(topology[l + 1]);
    }
  }

  // ── Forward pass ──────────────────────────────────────────────────────────
  // Feed the input through every layer in sequence.
  // We save z and a at every layer so backprop can use them.
  vctr predict(const vctr& input)
  {
    last_a[0] = input;  // treat the raw input as "activated output of layer -1"

    int num_layers = static_cast<int>(weights.size());
    for (int l = 0; l < num_layers; l++)
    {
      last_z[l] = mtrxMul(weights[l], last_a[l]);  // z = W * a_prev
      for (size_t i = 0; i < last_z[l].size(); i++)
        last_z[l][i] += biases[l][i];              // z += b

      last_a[l + 1] = relu(last_z[l]);             // a = LeakyReLU(z)
    }

    return last_a[num_layers];  // final layer's output
  }

  // ── Loss (MSE) ────────────────────────────────────────────────────────────
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

  // ── Backward pass ─────────────────────────────────────────────────────────
  // This is where multi-layer backprop differs from the single-layer version.
  //
  // In a single layer:  delta = (predicted - actual) * relu'(z)
  // In multiple layers: the output layer delta is the same, BUT for every
  //   hidden layer we must "send the error backwards" through the weight
  //   matrix to find out how much each hidden neuron contributed to the error.
  //
  // The chain rule gives us:
  //   delta[l] = (W[l+1]^T * delta[l+1]) * relu'(z[l])
  //              ─────────────────────    ───────────────
  //              error flowing back        local gate
  //
  // Once we have delta[l] for every layer, the weight update is the same as
  // before:  W[l] -= lr * delta[l] * a[l-1]^T
  void backPropagate(const vctr& actual)
  {
    int num_layers = static_cast<int>(weights.size());
    const vctr& predicted = last_a[num_layers];

    // Storage for deltas at each layer
    vector<vctr> deltas(num_layers);

    // ── Step 1: output layer delta ─────────────────────────────────────────
    // Same formula as the old single-layer code.
    {
      int L = num_layers - 1;
      vctr rd = reluDeriv(last_z[L]);
      deltas[L].resize(predicted.size());
      for (size_t i = 0; i < predicted.size(); i++)
        deltas[L][i] = (predicted[i] - actual[i]) * rd[i];
    }

    // ── Step 2: hidden layer deltas (going backwards through layers) ───────
    // We start at the second-to-last layer and walk toward layer 0.
    for (int l = num_layers - 2; l >= 0; l--)
    {
      vctr rd = reluDeriv(last_z[l]);
      int size_l = static_cast<int>(weights[l].size());           // neurons in THIS layer
      int size_l_next = static_cast<int>(weights[l + 1].size());  // neurons in NEXT layer

      deltas[l].assign(size_l, zero);

      // Multiply W[l+1]^T by delta[l+1]
      // W[l+1] is (size_l_next × size_l), so its transpose is (size_l × size_l_next)
      for (int i = 0; i < size_l; i++)
        for (int k = 0; k < size_l_next; k++)
          deltas[l][i] += weights[l + 1][k][i] * deltas[l + 1][k];

      // Multiply element-wise by relu derivative at this layer
      for (int i = 0; i < size_l; i++)
        deltas[l][i] *= rd[i];
    }

    // ── Step 3: update weights and biases for every layer ─────────────────
    // Identical pattern to before, just repeated for each layer.
    for (int l = 0; l < num_layers; l++)
    {
      int rows = static_cast<int>(weights[l].size());
      int cols = static_cast<int>(weights[l][0].size());
      for (int i = 0; i < rows; i++)
      {
        for (int j = 0; j < cols; j++)
          weights[l][i][j] -= learning_rate * deltas[l][i] * last_a[l][j];
        biases[l][i] -= learning_rate * deltas[l][i];
      }
    }
  }

  // ── Loss-based dynamic learning rate ──────────────────────────────────────
  // Call this at the end of each epoch with the current average loss.
  //
  // Logic:
  //   - We track the best loss seen so far.
  //   - If the new loss is better (lower) by at least `min_delta`, we record
  //     it and do nothing — training is still improving.
  //   - If loss has NOT improved by at least `min_delta` for `patience` epochs
  //     in a row, we multiply lr by `decay_factor` (e.g. 0.5) to slow down.
  //   - lr is clamped to a minimum floor so it never reaches zero.
  //
  // This is a simplified version of the classic "ReduceLROnPlateau" scheduler
  // used in PyTorch/Keras.
  void updateLearningRate(float current_loss)
  {
    constexpr int patience = 2;           // epochs to wait before decaying
    constexpr float min_delta = 1e-4F;    // minimum improvement to count as "better"
    constexpr float decay_factor = 0.5F;  // multiply lr by this when stuck
    constexpr float min_lr = 1e-5F;       // never go below this

    static float best_loss = 1e9F;
    static int epochs_no_improve = 0;

    if (current_loss < best_loss - min_delta)
    {
      best_loss = current_loss;
      epochs_no_improve = 0;
    }
    else
    {
      ++epochs_no_improve;
      if (epochs_no_improve >= patience)
      {
        learning_rate = max(learning_rate * decay_factor, min_lr);
        epochs_no_improve = 0;
        cout << "  [LR] No improvement for " << patience << " epochs → lr decayed to "
             << learning_rate << "\n";
      }
    }
  }

  // ── Save / Load ───────────────────────────────────────────────────────────
  // Same binary format as before, just repeated for each layer.
  void saveTraining(const string& filename)
  {
    ofstream file(filename, ios::binary);
    int num_layers = static_cast<int>(weights.size());
    file.write(reinterpret_cast<const char*>(&num_layers), sizeof(int));

    for (int l = 0; l < num_layers; l++)
    {
      int rows = static_cast<int>(weights[l].size());
      int cols = static_cast<int>(weights[l][0].size());
      file.write(reinterpret_cast<const char*>(&rows), sizeof(int));
      file.write(reinterpret_cast<const char*>(&cols), sizeof(int));
      for (int i = 0; i < rows; i++)
        file.write(reinterpret_cast<const char*>(weights[l][i].data()), cols * sizeof(float));

      int bsize = static_cast<int>(biases[l].size());
      file.write(reinterpret_cast<const char*>(&bsize), sizeof(int));
      file.write(reinterpret_cast<const char*>(biases[l].data()), bsize * sizeof(float));
    }

    // Also save the current learning rate so a resumed run continues smoothly
    file.write(reinterpret_cast<const char*>(&learning_rate), sizeof(float));

    cout << "Model saved to: " << filename << "\n";
  }

  bool loadTraining(const string& filename)
  {
    ifstream file(filename, ios::binary);
    if (!file.is_open())
    {
      cerr << "[WARNING] No saved model found. Starting fresh.\n";
      return false;
    }

    int num_layers;
    file.read(reinterpret_cast<char*>(&num_layers), sizeof(int));

    weights.resize(num_layers);
    biases.resize(num_layers);
    last_z.resize(num_layers);
    last_a.resize(num_layers + 1);

    for (int l = 0; l < num_layers; l++)
    {
      int rows, cols;
      file.read(reinterpret_cast<char*>(&rows), sizeof(int));
      file.read(reinterpret_cast<char*>(&cols), sizeof(int));

      weights[l].assign(rows, vctr(cols, zero));
      for (int i = 0; i < rows; i++)
        file.read(reinterpret_cast<char*>(weights[l][i].data()), cols * sizeof(float));

      int bsize;
      file.read(reinterpret_cast<char*>(&bsize), sizeof(int));
      biases[l].resize(bsize);
      file.read(reinterpret_cast<char*>(biases[l].data()), bsize * sizeof(float));
    }

    file.read(reinterpret_cast<char*>(&learning_rate), sizeof(float));

    cout << "Model loaded from: " << filename << " (lr=" << learning_rate << ")\n";
    return true;
  }

  // ── Evaluation ────────────────────────────────────────────────────────────
  int argmax(const vctr& v)
  {
    return static_cast<int>(max_element(v.begin(), v.end()) - v.begin());
  }

  void evaluate(const vector<vctr>& inputs, const vector<vctr>& actuals)
  {
    int correct = 0;
    int total = static_cast<int>(inputs.size());
    for (int i = 0; i < total; i++)
    {
      vctr output = predict(inputs[i]);
      if (argmax(output) == argmax(actuals[i]))
        ++correct;
    }
    float accuracy = 100.0F * static_cast<float>(correct) / static_cast<float>(total);
    cout << "Accuracy: " << correct << "/" << total << "  (" << accuracy << "%)\n";
  }
};

// ─── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
  const string model_file = "new.bin";

  // Topology: input=784, hidden1=256, hidden2=128, hidden3=64, output=10
  // To change the architecture just edit this list — the rest adapts automatically.
  const vector<int> topology = {784, 256, 128, 64, 32, 16, 10};

  ANN ann(topology, 0.01F);  // start with lr=0.01; will decay automatically

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
        ann.backPropagate(actuals[idx]);  // note: no longer needs predicted as arg
                                          // since it reads last_a internally
      }

      float avg_loss = total_loss / static_cast<float>(n);
      cout << "Epoch " << epoch << "/" << epochs << " | Avg Loss: " << avg_loss << "\n";

      // Check if we should decay the learning rate based on this epoch's loss
      ann.updateLearningRate(avg_loss);

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