import cli.app
from PaintByNumbers.PictureGenerator import PerformKMeans


@cli.app.CommandLineApp
def GeneratePaintByNumber(app):
    print("Paint by numbers!")
    params = app.params
    PerformKMeans(params.path, params.result, params.numColors, params.c, params.r, params.n, params.o, params.addTransparency)
 
#ls.add_param("-l", "--long", help="list in long format", default=False, action="store_true")
GeneratePaintByNumber.add_param("--path", help="Path to picture", type=str)
GeneratePaintByNumber.add_param("--result", help="Path to result", type=str)
GeneratePaintByNumber.add_param("--numColors", help="Number of colors", type=int)
GeneratePaintByNumber.add_param("-r", help="Show the final result", default=False, action="store_true")
GeneratePaintByNumber.add_param("-c", help="Show the color guide", default=False, action="store_true")
GeneratePaintByNumber.add_param("-n", help="Show the color guide numbers", default=False, action="store_true")
GeneratePaintByNumber.add_param("-o", help="Show color outlines", default=False, action="store_true")
GeneratePaintByNumber.add_param("--addTransparency", help="Add transparency %", default=1.0, type=float)

if __name__ == "__main__":
    GeneratePaintByNumber.run()