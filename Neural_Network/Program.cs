namespace Neural_Network
{
    internal class Program
    {
        static void Main(string[] args)
        {
            //Program.cs is the same as orginal code on slides
            double[,] training_input = new double[,] { { 0, 0, 1 }, { 1, 1, 1 }, { 1, 0, 1 }, { 0, 1, 1 } };
            double[,] training_output = Matrix.Transpose(new double[,] { { 0, 1, 1, 0 } });

            var nn = new NN(training_input, training_output);

            Console.WriteLine("Weights before training:");
            Matrix.Print(nn.Weights);

            nn.Train();

            Console.WriteLine("Weights after training:");
            Matrix.Print(nn.Weights);

            double[,] output = nn.Predict(new double[,] { { 1, 0, 0 } });
            Console.WriteLine("Predict [1, 0, 0] => " + output[0, 0]);
        }
    }
}