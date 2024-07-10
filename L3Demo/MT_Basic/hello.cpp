#include <iostream>
#include <thread>

void DisplayGreeting()
{
    std::cout << "Hello, World" << std::endl;
}

int main()
{
    std::thread t(DisplayGreeting);
    t.join();
}
