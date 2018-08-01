import sys
import os
import datetime

class Anon:
    """ 'Bunch' dictionary wrapper improving d['fred'] syntax to d.fred (anonymous types)
        d = [Anon(card_value=card+1, suit_value=suit) for card in range(13) for suit in range(4)]
    """
    def __init__(self, **kwds):
            self.__dict__.update(kwds)

def class_face(instance):
    """ show interface components of class instance based on _private naming convention
        see dir(obj) for imported objects
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
                
def test():
    print "Util Unit Tests"
    print "==============="
    
    print "\n" \
          "Anon type Test 01"
    print "================="
    car=Anon(Colour="Red", Make="Ford")
    print car.Make
    
    print "\n"  \
          "Class_face Test 01"
    print "=================="
    class_face(car)
    
    print "\n" \
          "Anon type Test 02"
    print "================="
    d = [Anon(card_value=card+1, suit_value=suit) for card in range(13) for suit in range(4)]
    class_face(d[0])
    
    print "\n" \
          "Dev Info"
    print "========"
    
    print_args = {'pypath': sys.executable}
    print "Python Path: {pypath}".format(**print_args)
    print "  expecting env in anaconda hierarchy like C:\Users\Bob\Anaconda2\envs\Scikitlearn01\pythonw.exe"
    print "  otherwise activate anaconda environment (conda info --env, conda activate Scikitlearn01, conda info)"
    print "    and set (wing) project\project property\python executable to it eg c:\Anaconda2\envs\Scikitlearn01\python.exe"
    
    print "\n" \
          "Current working diectory: {}".format(os.getcwd())
    
    print "\n" \
          "Current DateTime: {0}".format(datetime.datetime.now())
    
    print "\n" \
          "Generally: dir(os), help(os), help(os.getenv), help(), help('FORMATTING')"
    
if __name__  == "__main__":
    test()
