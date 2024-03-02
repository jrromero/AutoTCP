from multiprocessing import Process
from multiprocessing.pool import Pool

class NoDaemonProcess(Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool to make use of the custom Process class
# so child processes can create new processes
class EvaluationPool(Pool):
    def Process(self, *args, **kwds):
        proc = super(EvaluationPool, self).Process(*args, **kwds)
        proc.__class__ = NoDaemonProcess
        return proc