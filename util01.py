import sys

class Anon:
    """ 'Bunch' dictionary wrapper improving d['fred'] syntax to d.fred
        d = [Anon(card_value=card+1, suit_value=suit) for card in range(13) for suit in range(4)]
    """
    def __init__(self, **kwds):
            self.__dict__.update(kwds)

def class_face(instance):
    """ show interface components of class instance based on _private naming convention
    """
    print
    print type(instance).__name__

    d = {}
    d.update(instance.__class__.__dict__)
    d.update(instance.__dict__)
    
    for k,v in sorted(d.items()):
        if not k.startswith('_'):
            _type_name = type(v).__name__ 
            _var_name = k
            if _type_name == "function":
                _var_name += "()"
            if _type_name in ["list", "function", "property"]:
                print _type_name.rjust(10), ":", _var_name
            else:
                print _type_name.rjust(10), ":", _var_name, "=", v
                
def python_path():
    print(sys.executable)
                
def test():
    print "Util Unit Tests"
    print "==============="
    print "Anon type Test 01"
    car=Anon(Colour="Red", Make="Ford")
    print car.Make
    
    print "class_face Test 01"
    class_face(car)
    
    print "Anon type Test 02"
    d = [Anon(card_value=card+1, suit_value=suit) for card in range(13) for suit in range(4)]
    class_face(d[0])
    
    print "Current Python Path:"
    python_path()
    print "expecting env in anaconda hierarchy like C:\Users\Bob\Anaconda2\envs\Scikitlearn01\pythonw.exe"
    print "otherwise activate anaconda environment (conda info --env, conda activate Scikitlearn01, conda info)"
    print "  and set (wing) project\project property\python executable to it eg c:\Anaconda2\envs\Scikitlearn01\python.exe"
    
if __name__  == "__main__":
    test()
