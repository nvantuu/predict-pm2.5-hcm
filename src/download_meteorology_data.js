const axios = require('axios')
const {JSDOM} = require("jsdom");
const ObjectsToCsv = require('objects-to-csv')

var getDaysArray = function(s,e) {for(var a=[],d=new Date(s);d<=e;d.setDate(d.getDate()+1)){ a.push(new Date(d));}return a;};
// Get all date of 3 year
var datelist = getDaysArray(new Date("2021-01-01"), new Date("2021-12-31"))
console.log(datelist.length)

process.env['NODE_TLS_REJECT_UNAUTHORIZED'] = '0';
for (let date of datelist) {
    // Create date query name following convention yyyymmdd
    let fname = date.toISOString().split('T')[0].replaceAll('-', '');
    const query = `https://weather.uwyo.edu/cgi-bin/wyowx.fcgi?TYPE=sflist&DATE=${fname}&HOUR=current&UNITS=M&STATION=VVTS`
    axios
        .get(query)
        .then(res => {
            // Get table data of specific date (fname)
            let data = "<!DOCTYPE html>" + res.data + "<\/BODY>\n<\/HTML>";
            const dom = new JSDOM(data)
            data = dom.window.document.querySelector("pre").textContent.split('\n');

            // Final space array is position of first character of each word in feature names
            let space = []
            for (let s of data[2].split(' ')) {
                space.push(s.length);
            }
            for (let i = 1; i < space.length; i++) {
                space[i] += space[i-1]+1
            }

            // Create list of meteorology data of this date, each hour is a record, record 1, 2 is header of table
            let objs = []
            for (let i = 3; i < data.length; i++) {
                // console.log(data[i])
                if (data[i].length < 1) continue;
                // prev is position of first character of prev word
                let prev = 0;
                let obj = {};
                // idx is list of feature name with corresponding unit (<feature name> (<unit>)
                let idx = [];
                // Count is updated number of empty feature names
                let count = 0;
                // Add all feature following feature name specific by space array
                for (let j = 0; j < space.length; j++) {
                    let annotation = null;
                    if (j === space.length - 1) {
                        annotation = `${data[0].substr(prev).trim()}`;
                    } else {
                        annotation = `${data[0].substr(prev, space[j] - prev).trim()}`;
                    }
                    // Add unit for this feature name
                    if (annotation === '') {
                        annotation = "NA_" + count;
                        count++;
                    }
                    if (j === space.length - 1) {
                        annotation += `(${data[1].substr(prev).trim()})`;
                    } else {
                        annotation += `(${data[1].substr(prev, space[j] - prev).trim()})`;
                    }
                    idx.push(annotation);
                    prev = space[j]+1;
                }
                // Add feature value for each feature in current hour
                prev = 0;
                for (let j = 0; j < space.length; j++) {
                    if (j === space.length - 1) {
                        obj[idx[j]] = data[i].substr(prev).trim();
                    } else {
                        obj[idx[j]] = data[i].substr(prev, space[j]-prev).trim();
                    }
                    prev = space[j]+1;
                }
                // Final data of 1 hour
                objs.push(obj)
            }
            // Save data of all hour in this date
            data = objs;
            const csv = new ObjectsToCsv(data)
            csv.toDisk(`../data/${fname}.csv`);
        })
        .catch(error => {
            console.error(error)
        })
}