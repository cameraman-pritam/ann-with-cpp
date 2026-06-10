#include <crow.h>
#include <crow/app.h>
#include <iostream>
using namespace std;

int main() {
  cout << "Starting app\n";

  crow::SimpleApp app;

  CROW_ROUTE(app, "/")([]() { return "Hello, World!"; });

  app.port(5173).multithreaded().run();
}