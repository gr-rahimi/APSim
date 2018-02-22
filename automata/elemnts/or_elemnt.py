from ste import S_T_E,StartType

class OrElement(S_T_E):

    def __init__(self, id):
        super(OrElement, self).__init__(start_type=StartType.non_start, is_report = False, is_marked = False,
                                        id = id, symbol_set= None)



