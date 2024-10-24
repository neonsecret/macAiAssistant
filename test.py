import rumps


def infer():
    print("test")


class AwesomeStatusBarApp(rumps.App):
    def __init__(self):
        super(AwesomeStatusBarApp, self).__init__("infer")
        self.menu = ["Run"]

    @rumps.clicked("Run")
    def prefs(self, _):
        infer()
        rumps.alert("I love Saniochichek!")


if __name__ == '__main__':
    AwesomeStatusBarApp().run()
