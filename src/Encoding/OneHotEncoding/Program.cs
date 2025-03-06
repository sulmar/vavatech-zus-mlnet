
using Microsoft.ML;

Console.WriteLine("Hello, One Hot Encoding!");

var context = new MLContext();

var data = new CategoryData[]
{
    new CategoryData { Category = "kot"},
    new CategoryData { Category = "pies"},
    new CategoryData { Category = "ryba"},
    new CategoryData { Category = "kaczka"},
    new CategoryData { Category = "pies"},
};


var dataView = context.Data.LoadFromEnumerable(data);

var pipeline = context.Transforms.Categorical.OneHotEncoding("EncodedCategory", "Category");

var transformer = pipeline.Fit(dataView);
var transformedData = transformer.Transform(dataView);

var encodedData = context.Data.CreateEnumerable<CategoryEncoded>(transformedData, reuseRowObject: false);


foreach (var item in encodedData)
{
    Console.WriteLine($"Category: {item.Category} One hot encoding: [ {string.Join(", ",  item.EncodedCategory)} ]");
}

Console.ReadLine();

public class CategoryData
{
    public string Category { get; set; }
}


public class CategoryEncoded : CategoryData
{
    public float[] EncodedCategory { get; set; }
}
