import cli.app
from PaintByNumbers.PictureGenerator import PerformKMeans


@cli.app.CommandLineApp
def GeneratePaintByNumber(app):
    print("Paint by numbers!")
    params = app.params
    PerformKMeans(params.path, params.result, params.numColors, params.showColors, params.showResult, params.showNumbers, params.showOutlines, params.addTransparency)
 
#ls.add_param("-l", "--long", help="list in long format", default=False, action="store_true")
GeneratePaintByNumber.add_param("--path", help="Path to picture", type=str)
GeneratePaintByNumber.add_param("--result", help="Path to result", type=str)
GeneratePaintByNumber.add_param("--numColors", help="Number of colors", type=int)
GeneratePaintByNumber.add_param("--showResult", help="Show the final result", default=True, type=bool)
GeneratePaintByNumber.add_param("--showColors", help="Show the color guide", default=True, type=bool)
GeneratePaintByNumber.add_param("--showNumbers", help="Show the color guide numbers", default=True, type=bool)
GeneratePaintByNumber.add_param("--showOutlines", help="Show color outlines", default=False, type=bool)
GeneratePaintByNumber.add_param("--addTransparency", help="Add transparency %", default=1.0, type=float)

if __name__ == "__main__":
    GeneratePaintByNumber.run()