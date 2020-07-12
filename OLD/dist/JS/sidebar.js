var SidebarMenuEffects = (function () {

    function hasParentClass(e, classname) {
        if (e === document) return false;
        if (classie.has(e, classname)) {
            return true;
        }
        return e.parentNode && hasParentClass(e.parentNode, classname);
    }


    function initSide() {
        var container = document.getElementById('st-container'),
            buttons = Array.prototype.slice.call(document.querySelectorAll('#st-trigger-effects > button')),
            // event type (if mobile use touch events)
            eventtype = 'click',
            resetMenu = function () {
                classie.remove(container, 'st-menu-open');
            },
            bodyClickFn = function (evt) {
                if (!hasParentClass(evt.target, 'st-menu')) {
                    resetMenu();
                    document.removeEventListener(eventtype, bodyClickFn);
                }
            };

        buttons.forEach(function (el, i) {
            var effect = el.getAttribute('data-effect');
            el.addEventListener(eventtype, function (ev) {
                ev.stopPropagation();
                ev.preventDefault();
                container.className = 'st-container'; // clear
                classie.add(container, effect);
                setTimeout(function () {
                    classie.add(container, 'st-menu-open');
                }, 25);
                document.addEventListener(eventtype, bodyClickFn);
            });
        });

    }

    initSide();

})();
