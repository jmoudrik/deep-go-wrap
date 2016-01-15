import re

from collections import namedtuple

BrWr = namedtuple('BrWr', 'br wr')

class RankInitExc(Exception):
    pass

def argmin(pairs):
    return min(pairs, key=lambda x:x[1])[0]

class Rank:
    KEYS={'k': lambda x: x,         # 1kyu -> 1, 30kyu -> 30
          'd': lambda x: -x + 1,    # 1dan -> 0, 10dan -> -9
          'p': lambda x: -x - 9}    # 1pro -> -10, 10pro -> -19

    DOMAIN_MAX = { 'k' : 30,
                   'd' : 10,
                   'p' : 10 }

    @staticmethod
    def from_key(number):
        # XXX ugly
        ranks = list(Rank.iter_all())
        dists = [ abs(number - r.key()) for r in ranks ]
        return argmin( zip(ranks, dists) )

    @staticmethod
    def from_string(string, strict=False):
        rexp = '^([1-9][0-9]?) ?([kdp]).*'
        if strict:
            rexp = '^([1-9][0-9]?) ?([kdp])$'
        mo = re.match(rexp, string.lower())
        if not mo:
            return None
        try:
            return Rank(int(mo.group(1)), mo.group(2))
        except (ValueError, RankInitExc):
            return None

    @staticmethod
    def iter_all():
        for key, domain in Rank.DOMAIN_MAX.iteritems():
            for x in xrange(domain):
                yield Rank( x + 1, key )

    def __init__(self, number, kdp):
        self.number, self.kdp = number, kdp

        if not self.kdp in self.KEYS:
            raise RankInitExc("kdp must be either 'k' for kyu players,"
                             " 'd' for dan players or 'p' for proffesionals")

        def check_domain(bottom, val, up):
            assert bottom <= up
            if not( bottom <= val <= up):
                raise RankInitExc("Must be %d <= %d <= %d.")

        check_domain(1, self.number, self.DOMAIN_MAX[self.kdp])

    def as_tuple(self):
        return self.number, self.kdp

    def key(self):
        return self.KEYS[self.kdp](self.number)

    def __str__(self):
        return "%d%s"%(self.number, self.kdp)

    def __repr__(self):
        return "Rank(%s, key=%d)"%(self, self.key())

    def __hash__(self):
        return self.key().__hash__()

    def __cmp__(self, other):
        if not isinstance(other, Rank):
            return -1
        return ( - self.key()).__cmp__( - other.key())


if __name__ == "__main__":

    assert Rank(6, 'd') >  Rank(2, 'd') > Rank(1, 'k') > Rank(10, 'k')

    def print_rank():
        print "["
        for rank in Rank.iter_all():
            value =  rank.key()
            text = str(rank)
            print '{"value" : "%s", "text" : "%s"},' % (value,  text)
        print "]"

    print_rank()

