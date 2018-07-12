import random
import util01

class Card(object):
    _display_values = {1: 'A', 11: 'J', 12: 'Q', 13: 'K'}
    _display_suits = {1:'C', 2: 'D', 3:'H', 4:'S'}
    fred = 42
    
    def __init__(self, card_id):
        self._card_id = card_id
        self.card_value = (card_id % 13) + 1
        self.suit_value = (card_id / 13) + 1
        self.display = "{display_value}{display_suit}".format(
            display_value = self._display_values.get(self.card_value, self.card_value),
            display_suit = self._display_suits.get(self.suit_value, '?'))
            
    def face_value(self):
        return self.display
        
    def info(self):
        return "{display_value}{display_suit} {card_id} v:{card_value} s:{suit_value}".format(
            display_value = self._display_values.get(self.card_value, self.card_value),
            display_suit = self._display_suits.get(self.suit_value, '?'),
            card_id = self._card_id,
            card_value = self.card_value,
            suit_value = self.suit_value)
            
    def __str__(self):
        return (str(self._display_values.get(self.card_value, self.card_value)) +
                str(self._display_suits.get(self.suit_value, '?')))

    @property
    def condition(self):
        return self._condition
    
    @condition.setter
    def condition(self, condition):
        if condition == "Marked":
            raise ValueError("Cheat!!")
        self._condition = condition

# add atttibutes to instance (not vars in class scope) same as c.card_value = 10
class Deck(object):
    def __init__(self): 
        self.deck = [Card(card_id) for card_id in range(52)]
        
    def shuffle(self):
        #for card in self.deck:
        #    card.sort_id = random.random()
        #self.deck.sort(key=card.sort_id)
        #random.shuffle(self.deck
        pass

    def show(self):
        return ', '.join([card.face_value() for card in self.deck])
        
    def show2(self):
        for card in self.deck:
            print(card)

def main():
    c1 = Card(39)
    util01.ClassFace(Card(39))
    util01.ClassFace(Deck())
    print "hi2"

    #dir(c1)
    #help(inspect.getmembers())
    #print(c1.__class__.__dict__)
    #p = [k, type(v) for k,v in c1.__class__.__dict__.items() if not k.startswith('_')]
#    for k,v in c1.__class__.__dict__.items():
#        if not k.startswith('_'):
#            print(k, type(v))
#    
#    print(p)
#    print(c1)
#    print(c1.info())
#    print(c1.display)
#    print(c1.__dict__)
#    print(c1.__class__.__dict__)
#    print(type(c1.__class__.__dict__))
#    #for k,v in 
#    c1.condition = "sticky"
#    print(c1.condition)
#    dir(c1)
#    deck=Deck()      
#    print(deck.show())
#
#    deck.shuffle()
#    print(deck.show())

    #deck.show2()
    
if __name__  == "__main__":
    main()
